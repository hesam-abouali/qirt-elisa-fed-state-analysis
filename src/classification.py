import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl

# Ensure high-quality saved figures
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial']

class EnhancedMultiModelClassifier:
    """
    Enhanced classifier comparing Logistic Regression, Random Forest, SVM, and KNN.
    
    Maintains same feature selection and processing pipeline while adding model comparison.
    Creates comprehensive visualization comparing all four models.
    """
    
    def __init__(self):
        # Same top 15 features as original (LR_Coef importance order)
        self.top_15_features = [
            'glucagon_kinetics_mean_rise_rate',      # LR=0.301, F=14.82
            'insulin_kinetics_mean_rise_time',       # LR=0.262, F=7.80  
            'insulin_kinetics_peak_count',           # LR=0.254, F=inf
            'glucagon_elisa_concentration_range_pM', # LR=0.234, F=7.94
            'glucagon_elisa_calibration_slope',      # LR=0.197, F=5.12
            'glucagon_elisa_max_secretion_rate',     # LR=0.192, F=6.14
            'insulin_peak_count_above_std',          # LR=0.191, F=6.00
            'glucagon_kinetics_peak_count',          # LR=0.187, F=19.00
            'insulin_osc_rhythmicity_index',         # LR=0.186, F=12.83
            'c_peptide_osc_rhythmicity_index',       # LR=0.176, F=6.81
            'glucagon_kinetics_peak_frequency',      # LR=0.171, F=inf
            'glucagon_kinetics_rise_decline_ratio',  # LR=0.168, F=12.10
            'c_peptide_osc_signal_entropy',          # LR=0.156, F=9.41
            'glucagon_glucose_responsiveness',       # LR=0.151, F=5.08
            'insulin_osc_frequency'                  # LR=0.131, F=7.00
        ]
        
        # Original importance scores for reference
        self.lr_coefficients = [0.301, 0.262, 0.254, 0.234, 0.197, 0.192, 0.191, 
                               0.187, 0.186, 0.176, 0.171, 0.168, 0.156, 0.151, 0.131]
        
        # Model storage
        self.models = {}
        self.model_results = {}
        self.scaler = None
        self.data = None
        self.features_scaled = None
        self.features_raw = None
        self.labels = None
        self.animal_ids = None
        self.label_names = ['Healthy', 'Pre-diabetic', 'Diabetic']
        self.colors = ['#2E8B57', '#FF8C00', '#DC143C']  # Green, Orange, Red
        self.markers = ['o', 's', '^']
        self.plotted_top2_data = None
        
        print("Enhanced Multi-Model Fed State Classifier Initialized")
        print("Models: Logistic Regression, Random Forest, SVM, K-Nearest Neighbors")
        print("Same feature selection pipeline maintained")
    
    def load_and_prepare_data(self, csv_path):
        """Load data and prepare features (identical to original)"""
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load data
        self.data = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(self.data)} animals with {len(self.data.columns)} columns")
        
        # Check available features
        available_features = [f for f in self.top_15_features if f in self.data.columns]
        missing_features = [f for f in self.top_15_features if f not in self.data.columns]
        
        if missing_features:
            print(f"[WARNING] Missing {len(missing_features)} features:")
            for feat in missing_features[:3]:  # Show first 3
                print(f"   • {feat}")
            if len(missing_features) > 3:
                print(f"   • ... and {len(missing_features)-3} more")
        
        print(f"[OK] Using {len(available_features)}/15 available features")
        
        # Extract feature matrix (RAW VALUES)
        feature_matrix_raw = self.data[available_features].values
        feature_matrix_raw = np.nan_to_num(feature_matrix_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store raw values before scaling
        self.features_raw = feature_matrix_raw.copy()
        
        # Scale features
        self.scaler = RobustScaler()
        self.features_scaled = self.scaler.fit_transform(feature_matrix_raw)
        
        # Prepare labels and IDs
        self.animal_ids = self.data['animal_id'].values
        label_mapping = {'healthy': 0, 'pre-diabetic': 1, 'diabetic': 2}
        self.labels = self.data['rat_type'].map(label_mapping).values
        
        # Store available features info
        self.available_features = available_features
        self.n_features_used = len(available_features)
        
        print(f"[OK] Feature matrix shape: {self.features_scaled.shape}")
        print(f"[OK] Groups: {dict(zip(*np.unique(self.data['rat_type'], return_counts=True)))}")
        
        return available_features
    
    def train_all_models(self, hyperparameter_tuning=True):
        """Train all four models with optional hyperparameter tuning"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        # Define models with small-sample optimizations
        base_models = {
            'Logistic Regression': LogisticRegression(
                penalty='l2', C=1.0, max_iter=1000, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50, max_depth=5, min_samples_leaf=2, 
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, 
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=3, weights='distance'
            )
        }
        
        # Hyperparameter grids for tuning (excluding LR to match original)
        param_grids = {
            'Random Forest': {
                'n_estimators': [20, 50],
                'max_depth': [3, 5],
                'min_samples_leaf': [2, 3]
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'KNN': {
                'n_neighbors': [3, 5],
                'weights': ['uniform', 'distance']
            }
        }
        
        # Cross-validation setup
        cv = LeaveOneOut()
        
        # Train each model
        for model_name, base_model in base_models.items():
            print(f"\n[Training] {model_name}...")
            
            try:
                if hyperparameter_tuning and model_name in param_grids:
                    # Hyperparameter tuning (except for LR which uses original params)
                    grid_search = GridSearchCV(
                        base_model, param_grids[model_name], 
                        cv=3, scoring='accuracy', n_jobs=1
                    )
                    grid_search.fit(self.features_scaled, self.labels)
                    best_model = grid_search.best_estimator_
                    print(f"   • Best params: {grid_search.best_params_}")
                else:
                    # Use default parameters (always for LR to match original)
                    best_model = base_model
                    best_model.fit(self.features_scaled, self.labels)
                    if model_name == 'Logistic Regression':
                        print(f"   • Using original parameters: C=1.0, penalty='l2'")
                    else:
                        print(f"   • Using default parameters")
                
                # Cross-validation evaluation
                cv_predictions = []
                cv_probabilities = []
                
                for train_idx, test_idx in cv.split(self.features_scaled):
                    X_train, X_test = self.features_scaled[train_idx], self.features_scaled[test_idx]
                    y_train, y_test = self.labels[train_idx], self.labels[test_idx]
                    
                    # Train fold model
                    if hyperparameter_tuning and model_name in param_grids:
                        fold_grid = GridSearchCV(
                            base_models[model_name], param_grids[model_name],
                            cv=2, scoring='accuracy', n_jobs=1  # Reduced CV for LOO
                        )
                        fold_grid.fit(X_train, y_train)
                        fold_model = fold_grid.best_estimator_
                    else:
                        fold_model = base_models[model_name]
                        fold_model.fit(X_train, y_train)
                    
                    # Predict
                    pred = fold_model.predict(X_test)[0]
                    cv_predictions.append(pred)
                    
                    # Get probabilities if available
                    if hasattr(fold_model, 'predict_proba'):
                        prob = fold_model.predict_proba(X_test)[0]
                        cv_probabilities.append(prob)
                    else:
                        # For models without predict_proba, use decision function or dummy probs
                        dummy_prob = np.zeros(3)
                        dummy_prob[pred] = 1.0
                        cv_probabilities.append(dummy_prob)
                
                # Training predictions
                training_predictions = best_model.predict(self.features_scaled)
                if hasattr(best_model, 'predict_proba'):
                    training_probabilities = best_model.predict_proba(self.features_scaled)
                else:
                    training_probabilities = np.eye(3)[training_predictions]
                
                # Calculate metrics
                training_accuracy = accuracy_score(self.labels, training_predictions)
                cv_accuracy = accuracy_score(self.labels, cv_predictions)
                avg_confidence = np.mean([np.max(prob) for prob in cv_probabilities])
                
                # Store results
                self.models[model_name] = best_model
                self.model_results[model_name] = {
                    'training_accuracy': training_accuracy,
                    'cv_accuracy': cv_accuracy,
                    'avg_confidence': avg_confidence,
                    'training_predictions': training_predictions,
                    'cv_predictions': cv_predictions,
                    'training_probabilities': training_probabilities,
                    'cv_probabilities': cv_probabilities
                }
                
                print(f"   • Training accuracy: {training_accuracy:.3f}")
                print(f"   • CV accuracy: {cv_accuracy:.3f}")
                print(f"   • Avg confidence: {avg_confidence:.3f}")
                
            except Exception as e:
                print(f"   [ERROR] Failed to train {model_name}: {str(e)}")
        
        # Print summary
        print(f"\nMODEL COMPARISON SUMMARY:")
        print("-" * 60)
        print(f"{'Model':<20} | {'Training Acc':<12} | {'CV Acc':<8} | {'Gap':<8}")
        print("-" * 60)
        
        for model_name in self.model_results:
            result = self.model_results[model_name]
            gap = result['training_accuracy'] - result['cv_accuracy']
            print(f"{model_name:<20} | {result['training_accuracy']:<12.3f} | {result['cv_accuracy']:<8.3f} | {gap:<8.3f}")
        
        # Best model
        best_cv_model = max(self.model_results.keys(), 
                           key=lambda x: self.model_results[x]['cv_accuracy'])
        print(f"\n[Best] Best CV Performance: {best_cv_model} ({self.model_results[best_cv_model]['cv_accuracy']:.3f})")
    
    def create_enhanced_visualization(self, save_path=None):
        """Create enhanced visualization comparing all models"""
        print("\n" + "="*60)
        print("CREATING ENHANCED MULTI-MODEL VISUALIZATION")
        print("="*60)
        
        # Create figure with more subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('QIRT-ELISA Biomarker Dynamics Analysis - Fed State Experiments',
                    fontsize=16, y=0.95)
        
        # Plot 2: Decision Boundaries Comparison (2x2 grid in first 2x2 positions)
        self._plot_decision_boundaries_comparison(fig, gs)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Enhanced visualization saved to: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_model_comparison(self, ax):
        """Plot model performance comparison"""
        models = list(self.model_results.keys())
        training_accs = [self.model_results[model]['training_accuracy'] for model in models]
        cv_accs = [self.model_results[model]['cv_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, training_accs, width, label='Training Accuracy', 
                      color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, cv_accs, width, label='CV Accuracy', 
                      color='orange', alpha=0.7)
        
        # Add value labels
        for bar, acc in zip(bars1, training_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.2f}', ha='center', va='bottom', fontsize=10)
        
        for bar, acc in zip(bars2, cv_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison\n(Training vs Cross-Validation)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=10)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_decision_boundaries_comparison(self, fig, gs):
        """Plot decision boundaries for all models using top 2 features"""
        # Use top 2 features
        top2_features = self.features_scaled[:, :2]
        
        # Store data for reporting
        self.plotted_top2_data = {
            'animal_ids': self.animal_ids,
            'labels': self.labels,
            'features_scaled': top2_features,
            'features_raw': self.features_raw[:, :2],
            'feature_names': [self.available_features[0], self.available_features[1]]
        }
        
        model_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, (model_name, model) in enumerate(self.models.items()):
            row, col = model_positions[i]
            ax = fig.add_subplot(gs[row, col])
            
            # Create decision boundary
            h = 0.02
            x_min, x_max = top2_features[:, 0].min() - 1, top2_features[:, 0].max() + 1
            y_min, y_max = top2_features[:, 1].min() - 1, top2_features[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Train model on just top 2 features for boundary visualization
            if model_name == 'Logistic Regression':
                # Use exact same parameters as original code
                temp_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
            else:
                temp_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            temp_model.fit(top2_features, self.labels)
            
            # Predict on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = temp_model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary regions
            cmap = ListedColormap(['#E8F5E8', '#FFF4E6', '#FFE4E1'])
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
            ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.7)
            
            # Plot data points
            for j, (label, color, marker) in enumerate(zip(self.label_names, self.colors, self.markers)):
                mask = self.labels == j
                if np.any(mask):
                    ax.scatter(top2_features[mask, 0], top2_features[mask, 1], 
                              c=color, marker=marker, s=100, alpha=0.9, 
                              edgecolors='black', linewidths=1.5, 
                              label=f'{label} (n={np.sum(mask)})')
            
            # Get accuracy for this model on top 2 features
            top2_accuracy = accuracy_score(self.labels, temp_model.predict(top2_features))
            cv_accuracy = self.model_results[model_name]['cv_accuracy']
            
            ax.set_xlabel('Glucagon Kinetics Mean Rise Rate', fontsize=10)
            ax.set_ylabel('Insulin Kinetics Mean Rise Time', fontsize=10)
            ax.set_title(f'{model_name}\nTop2: {top2_accuracy:.1%} | CV: {cv_accuracy:.1%}', 
                        fontsize=11)
            if i == 0:  # Only show legend for first plot
                ax.legend(loc='best', fontsize=8)
    
    def create_15d_visualization(self, save_path=None):
        """Create visualization showing 15-dimensional classification results"""
        print("\n" + "="*60)
        print("CREATING 15-DIMENSIONAL CLASSIFICATION VISUALIZATION")
        print("="*60)
        
        # Create figure for 15D analysis
        fig = plt.figure(figsize=(10, 8))
        
        fig.suptitle('Fed State Classification: 15-Dimensional Analysis\n'
                    'LR vs SVM Performance on Complete Feature Set', 
                    fontsize=16, y=0.95)
        
        # Plot 6: Classification uncertainty heatmap
        ax6 = fig.add_subplot(1, 1, 1)
        self._plot_classification_uncertainty(ax6)
        
        plt.tight_layout()
        
        plt.show()
        return fig
    
    def _plot_pca_with_decision_boundary(self, fig, gs, model_name, position):
        """Plot PCA space with decision boundaries from 15D model"""
        ax = fig.add_subplot(gs[position[0], position[1]])
        
        # Apply PCA to all 15 features
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(self.features_scaled)
        
        # Get the trained model
        model = self.models[model_name]
        
        # Create decision boundary in PCA space
        h = 0.1
        x_min, x_max = pca_features[:, 0].min() - 1, pca_features[:, 0].max() + 1
        y_min, y_max = pca_features[:, 1].min() - 1, pca_features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Transform mesh points back to original 15D space for prediction
        mesh_pca = np.c_[xx.ravel(), yy.ravel()]
        
        # Since we can't perfectly inverse transform PCA, we'll approximate
        # by finding the closest points in PCA space and using their 15D values
        mesh_predictions = []
        for point in mesh_pca:
            # Find closest actual data point in PCA space
            distances = np.sum((pca_features - point)**2, axis=1)
            closest_idx = np.argmin(distances)
            # Use the 15D features of the closest point for prediction
            pred = model.predict(self.features_scaled[closest_idx:closest_idx+1])[0]
            mesh_predictions.append(pred)
        
        Z = np.array(mesh_predictions).reshape(xx.shape)
        
        # Plot decision boundary regions
        cmap = ListedColormap(['#E8F5E8', '#FFF4E6', '#FFE4E1'])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.7, levels=[0.5, 1.5])
        
        # Get model results
        model_result = self.model_results[model_name]
        predictions = model_result['cv_predictions']
        probabilities = model_result['cv_probabilities']
        
        # Plot data points with prediction confidence
        for i in range(len(self.label_names)):
            true_mask = self.labels == i
            if np.any(true_mask):
                # Get confidence for this class
                confidences = [prob[i] if len(prob) > i else 0 for prob in probabilities]
                confidence_colors = np.array(confidences)[true_mask]
                
                scatter = ax.scatter(pca_features[true_mask, 0], pca_features[true_mask, 1], 
                          c=confidence_colors, cmap='viridis', 
                          marker=self.markers[i], s=150, alpha=0.8,
                          edgecolors='black', linewidths=2, 
                          vmin=0, vmax=1)
                
                # Add animal IDs for misclassified points
                for j, animal_id in enumerate(self.animal_ids[true_mask]):
                    if predictions[np.where(true_mask)[0][j]] != i:
                        ax.annotate(f'#{animal_id}', 
                                  (pca_features[true_mask, 0][j], pca_features[true_mask, 1][j]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=10, color='red', weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Prediction Confidence', fontsize=10)
        
        cv_acc = model_result['cv_accuracy']
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
        ax.set_title(f'{model_name}\n15D CV Accuracy: {cv_acc:.1%}', fontsize=12)
    
    def _plot_prediction_confidence_comparison(self, ax):
        """Compare prediction confidence between LR and SVM"""
        lr_conf = [max(prob) for prob in self.model_results['Logistic Regression']['cv_probabilities']]
        svm_conf = [max(prob) for prob in self.model_results['SVM']['cv_probabilities']]
        
        x = np.arange(len(self.animal_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lr_conf, width, label='Logistic Regression', 
                      color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, svm_conf, width, label='SVM', 
                      color='red', alpha=0.7)
        
        # Color bars by correctness
        lr_correct = self.model_results['Logistic Regression']['cv_predictions'] == self.labels
        svm_correct = self.model_results['SVM']['cv_predictions'] == self.labels
        
        for i, (bar1, bar2, lr_c, svm_c) in enumerate(zip(bars1, bars2, lr_correct, svm_correct)):
            if not lr_c:
                bar1.set_color('lightcoral')
            if not svm_c:
                bar2.set_color('lightcoral')
        
        ax.set_xlabel('Animal ID', fontsize=11)
        ax.set_ylabel('Prediction Confidence', fontsize=11)
        ax.set_title('Prediction Confidence Comparison\n(Light red = Misclassified)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{aid}' for aid in self.animal_ids], rotation=45, fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_contribution_analysis(self, ax):
        """Analyze which features contribute most to classification differences"""
        # Compare LR coefficients with SVM decision function contributions
        lr_model = self.models['Logistic Regression']
        svm_model = self.models['SVM']
        
        # Get LR coefficients (absolute mean across classes)
        if hasattr(lr_model, 'coef_'):
            lr_importance = np.abs(np.mean(lr_model.coef_, axis=0))
        
        # For SVM, we'll use the magnitude of support vectors
        if hasattr(svm_model, 'coef_'):
            svm_importance = np.abs(np.mean(svm_model.coef_, axis=0))
        
        # Plot top 10 features
        top_indices = np.argsort(lr_importance)[-10:][::-1]
        top_features = [self.available_features[i].replace('_', ' ')[:25] for i in top_indices]
        lr_values = lr_importance[top_indices]
        svm_values = svm_importance[top_indices]
        
        y_pos = np.arange(len(top_features))
        
        bars1 = ax.barh(y_pos - 0.2, lr_values, 0.4, label='Logistic Regression', 
                       color='blue', alpha=0.7)
        bars2 = ax.barh(y_pos + 0.2, svm_values, 0.4, label='SVM', 
                       color='red', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Feature Importance: LR vs SVM\n(Top 10 Features)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_agreement_analysis(self, ax):
        """Analyze where LR and SVM agree/disagree"""
        lr_preds = self.model_results['Logistic Regression']['cv_predictions']
        svm_preds = self.model_results['SVM']['cv_predictions']
        
        # Create agreement matrix
        agreement_data = []
        for i in range(len(self.animal_ids)):
            agreement_data.append({
                'Animal': f'#{self.animal_ids[i]}',
                'True': self.label_names[self.labels[i]],
                'LR': self.label_names[lr_preds[i]],
                'SVM': self.label_names[svm_preds[i]],
                'Agreement': lr_preds[i] == svm_preds[i],
                'Both_Correct': (lr_preds[i] == self.labels[i]) and (svm_preds[i] == self.labels[i])
            })
        
        # Plot agreement patterns
        animals = [d['Animal'] for d in agreement_data]
        agreement = [d['Agreement'] for d in agreement_data]
        both_correct = [d['Both_Correct'] for d in agreement_data]
        
        colors = ['green' if bc else 'orange' if agree else 'red' 
                 for bc, agree in zip(both_correct, agreement)]
        
        bars = ax.bar(range(len(animals)), [1]*len(animals), color=colors, alpha=0.7)
        
        ax.set_xlabel('Animals', fontsize=11)
        ax.set_ylabel('Agreement Status', fontsize=11)
        ax.set_title('LR vs SVM Agreement Analysis', fontsize=11)
        ax.set_xticks(range(len(animals)))
        ax.set_xticklabels(animals, rotation=45, fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['', 'Models'])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Both Correct'),
                         Patch(facecolor='orange', alpha=0.7, label='Agree but Wrong'),
                         Patch(facecolor='red', alpha=0.7, label='Disagree')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_classification_uncertainty(self, ax):
        """Plot classification uncertainty heatmap"""
        # Create uncertainty matrix for each animal and each model
        models_to_compare = ['Logistic Regression', 'SVM']
        uncertainty_matrix = []
        
        for animal_idx in range(len(self.animal_ids)):
            row = []
            for model_name in models_to_compare:
                probs = self.model_results[model_name]['cv_probabilities'][animal_idx]
                # Uncertainty = 1 - max(probability) 
                uncertainty = 1 - max(probs)
                row.append(uncertainty)
            uncertainty_matrix.append(row)
        
        # Create heatmap
        im = ax.imshow(uncertainty_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(models_to_compare)))
        ax.set_xticklabels(models_to_compare, rotation=45, fontsize=10)
        ax.set_yticks(range(len(self.animal_ids)))
        ax.set_yticklabels([f'#{aid}' for aid in self.animal_ids], fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Classification Uncertainty', fontsize=10)
        
        # Add text annotations
        for i in range(len(self.animal_ids)):
            for j in range(len(models_to_compare)):
                text = ax.text(j, i, f'{uncertainty_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Classification Uncertainty\n(1 - Max Probability)', fontsize=11)
    
    def calculate_comprehensive_model_metrics(self):
        """Calculate all theoretical metrics from the paper framework"""
        print("\n" + "="*60)
        print("CALCULATING COMPREHENSIVE MODEL METRICS")
        print("="*60)
        
        from scipy import stats
        
        metrics_results = {}
        
        for model_name in self.model_results:
            model_result = self.model_results[model_name]
            cv_predictions = model_result['cv_predictions']
            cv_probabilities = model_result['cv_probabilities']
            
            # Calculate uncertainties and confidences
            uncertainties = []
            confidences = []
            correct_predictions = []
            
            for i in range(len(self.labels)):
                prob = cv_probabilities[i]
                confidence = max(prob)
                uncertainty = 1 - confidence
                correct = cv_predictions[i] == self.labels[i]
                
                uncertainties.append(uncertainty)
                confidences.append(confidence)
                correct_predictions.append(correct)
            
            # 2.5.2 Model Reliability Score
            reliability_score = np.mean([
                (1 - u) * c for u, c in zip(uncertainties, correct_predictions)
            ])
            
            # Calibration Consistency (confidence for correct vs incorrect)
            correct_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
            incorrect_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
            
            calibration_consistency = (
                np.mean(correct_confidences) - np.mean(incorrect_confidences) 
                if incorrect_confidences else np.mean(correct_confidences)
            )
            
            # Confidence-Accuracy Consistency Score (correlation)
            if len(set(correct_predictions)) > 1:  # Need variation in correctness
                ccc_score = np.corrcoef(confidences, correct_predictions)[0, 1]
            else:
                ccc_score = 1.0 if all(correct_predictions) else 0.0
            
            # 2.5.6 Composite Reliability Index
            cv_accuracy = model_result['cv_accuracy']
            mean_uncertainty = np.mean(uncertainties)
            cri_score = cv_accuracy * (1 - mean_uncertainty) * abs(ccc_score)
            
            metrics_results[model_name] = {
                'uncertainties': uncertainties,
                'confidences': confidences,
                'correct_predictions': correct_predictions,
                'reliability_score': reliability_score,
                'calibration_consistency': calibration_consistency,
                'ccc_score': ccc_score,
                'mean_uncertainty': mean_uncertainty,
                'cri_score': cri_score,
                'cv_accuracy': cv_accuracy
            }
        
        # 2.5.3 Pairwise Model Comparison (focus on LR vs SVM)
        if 'Logistic Regression' in metrics_results and 'SVM' in metrics_results:
            lr_metrics = metrics_results['Logistic Regression']
            svm_metrics = metrics_results['SVM']
            
            # Uncertainty Dominance Test
            lr_total_uncertainty = sum(lr_metrics['uncertainties'])
            svm_total_uncertainty = sum(svm_metrics['uncertainties'])
            uncertainty_dominance = 'LR' if lr_total_uncertainty < svm_total_uncertainty else 'SVM'
            
            # 2.5.7 Statistical Significance Testing
            # Wilcoxon signed-rank test for paired uncertainty differences
            uncertainty_diffs = np.array(svm_metrics['uncertainties']) - np.array(lr_metrics['uncertainties'])
            
            if len(uncertainty_diffs) >= 6:  # Minimum for Wilcoxon test
                try:
                    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(uncertainty_diffs, alternative='greater')
                except:
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            else:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            
            # Effect Size (Cohen's d for uncertainty)
            pooled_std = np.sqrt((np.var(lr_metrics['uncertainties']) + np.var(svm_metrics['uncertainties'])) / 2)
            cohens_d_uncertainty = (svm_metrics['mean_uncertainty'] - lr_metrics['mean_uncertainty']) / pooled_std if pooled_std > 0 else 0
            
            # Confidence difference test
            confidence_diffs = np.array(lr_metrics['confidences']) - np.array(svm_metrics['confidences'])
            if len(confidence_diffs) >= 6:
                try:
                    conf_wilcoxon_stat, conf_wilcoxon_p = stats.wilcoxon(confidence_diffs, alternative='greater')
                except:
                    conf_wilcoxon_stat, conf_wilcoxon_p = np.nan, np.nan
            else:
                conf_wilcoxon_stat, conf_wilcoxon_p = np.nan, np.nan
            
            pairwise_results = {
                'uncertainty_dominance': uncertainty_dominance,
                'lr_total_uncertainty': lr_total_uncertainty,
                'svm_total_uncertainty': svm_total_uncertainty,
                'uncertainty_wilcoxon_stat': wilcoxon_stat,
                'uncertainty_wilcoxon_p': wilcoxon_p,
                'cohens_d_uncertainty': cohens_d_uncertainty,
                'confidence_wilcoxon_stat': conf_wilcoxon_stat,
                'confidence_wilcoxon_p': conf_wilcoxon_p
            }
            
            metrics_results['pairwise_lr_svm'] = pairwise_results
        
        # 2.5.6 Model Ranking by CRI
        cri_ranking = sorted(
            [(name, metrics['cri_score']) for name, metrics in metrics_results.items() if 'cri_score' in metrics],
            key=lambda x: x[1], reverse=True
        )
        
        metrics_results['cri_ranking'] = cri_ranking
        
        self.comprehensive_metrics = metrics_results
        return metrics_results
    
    def create_comprehensive_metrics_visualization(self, save_path=None):
        """Create comprehensive visualization of all theoretical metrics"""
        if not hasattr(self, 'comprehensive_metrics'):
            self.calculate_comprehensive_model_metrics()
        
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE METRICS VISUALIZATION")
        print("="*60)
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        fig.suptitle('Comprehensive Model Assessment: Statistical Framework\n'
                    'LR vs SVM Performance Analysis with Theoretical Metrics', 
                    fontsize=18, y=0.95)
        
        # Plot 1: Model Reliability Scores
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_reliability_scores(ax1)
        
        # Plot 2: Calibration Consistency
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_calibration_consistency(ax2)
        
        # Plot 3: Confidence-Accuracy Correlation
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_confidence_accuracy_correlation(ax3)
        
        # Plot 4: Composite Reliability Index
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_composite_reliability_index(ax4)
        
        # Plot 5: Uncertainty Distribution Comparison
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_uncertainty_distributions(ax5)
        
        # Plot 6: Confidence Distribution Comparison  
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_confidence_distributions(ax6)
        
        # Plot 7: Statistical Significance Results
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_statistical_significance(ax7)
        
        # Plot 8: Effect Sizes
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_effect_sizes(ax8)
        
        # Plot 9: Individual Animal Uncertainty Comparison
        ax9 = fig.add_subplot(gs[2, :2])
        self._plot_individual_uncertainty_comparison(ax9)
        
        # Plot 10: Model Performance Summary Table
        ax10 = fig.add_subplot(gs[2, 2:])
        self._plot_performance_summary_table(ax10)
        
        plt.tight_layout()
        
        plt.show()
        return fig
    
    def _plot_model_reliability_scores(self, ax):
        """Plot Model Reliability Scores (R(h))"""
        models = ['Logistic Regression', 'SVM']
        scores = [self.comprehensive_metrics[m]['reliability_score'] for m in models if m in self.comprehensive_metrics]
        model_names = [m for m in models if m in self.comprehensive_metrics]
        
        bars = ax.bar(model_names, scores, color=['blue', 'red'], alpha=0.7)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Reliability Score R(h)', fontsize=12)
        ax.set_title('Model Reliability Assessment\nR(h) = (1-U_i) × Correct', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_calibration_consistency(self, ax):
        """Plot Calibration Consistency"""
        models = ['Logistic Regression', 'SVM'] 
        consistency = [self.comprehensive_metrics[m]['calibration_consistency'] for m in models if m in self.comprehensive_metrics]
        model_names = [m for m in models if m in self.comprehensive_metrics]
        
        bars = ax.bar(model_names, consistency, color=['blue', 'red'], alpha=0.7)
        
        for bar, cons in zip(bars, consistency):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{cons:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Calibration Consistency', fontsize=12)
        ax.set_title('Calibration Quality\nE[C_i|Correct] - E[C_i|Incorrect]', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_accuracy_correlation(self, ax):
        """Plot Confidence-Accuracy Consistency Score"""
        models = ['Logistic Regression', 'SVM']
        ccc_scores = [self.comprehensive_metrics[m]['ccc_score'] for m in models if m in self.comprehensive_metrics]
        model_names = [m for m in models if m in self.comprehensive_metrics]
        
        bars = ax.bar(model_names, ccc_scores, color=['blue', 'red'], alpha=0.7)
        
        for bar, ccc in zip(bars, ccc_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{ccc:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('CCC Score', fontsize=12)
        ax.set_title('Confidence-Accuracy Correlation\ncorr(Confidence, Correct)', fontsize=12)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_composite_reliability_index(self, ax):
        """Plot Composite Reliability Index"""
        models = ['Logistic Regression', 'SVM']
        cri_scores = [self.comprehensive_metrics[m]['cri_score'] for m in models if m in self.comprehensive_metrics]
        model_names = [m for m in models if m in self.comprehensive_metrics]
        
        bars = ax.bar(model_names, cri_scores, color=['blue', 'red'], alpha=0.7)
        
        for bar, cri in zip(bars, cri_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{cri:.3f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('CRI Score', fontsize=12)
        ax.set_title('Composite Reliability Index\nAcc × (1-U) × |CCC|', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_uncertainty_distributions(self, ax):
        """Plot uncertainty distributions for LR vs SVM"""
        lr_uncertainties = self.comprehensive_metrics['Logistic Regression']['uncertainties']
        svm_uncertainties = self.comprehensive_metrics['SVM']['uncertainties']
        
        ax.hist(lr_uncertainties, alpha=0.6, label='Logistic Regression', color='blue', bins=5)
        ax.hist(svm_uncertainties, alpha=0.6, label='SVM', color='red', bins=5)
        
        ax.set_xlabel('Classification Uncertainty', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Uncertainty Distribution Comparison', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_distributions(self, ax):
        """Plot confidence distributions for LR vs SVM"""
        lr_confidences = self.comprehensive_metrics['Logistic Regression']['confidences']
        svm_confidences = self.comprehensive_metrics['SVM']['confidences']
        
        ax.hist(lr_confidences, alpha=0.6, label='Logistic Regression', color='blue', bins=5)
        ax.hist(svm_confidences, alpha=0.6, label='SVM', color='red', bins=5)
        
        ax.set_xlabel('Prediction Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Confidence Distribution Comparison', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_significance(self, ax):
        """Plot statistical significance test results"""
        if 'pairwise_lr_svm' in self.comprehensive_metrics:
            pairwise = self.comprehensive_metrics['pairwise_lr_svm']
            
            tests = ['Uncertainty\nWilcoxon', 'Confidence\nWilcoxon']
            p_values = [pairwise['uncertainty_wilcoxon_p'], pairwise['confidence_wilcoxon_p']]
            
            # Handle NaN values
            p_values = [p if not np.isnan(p) else 1.0 for p in p_values]
            
            colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
            
            bars = ax.bar(tests, p_values, color=colors, alpha=0.7)
            
            for bar, p in zip(bars, p_values):
                significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{p:.3f}\n{significance}', ha='center', va='bottom', fontsize=10)
            
            ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='α = 0.05')
            ax.set_ylabel('p-value', fontsize=12)
            ax.set_title('Statistical Significance Tests\n(LR vs SVM)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_effect_sizes(self, ax):
        """Plot effect sizes"""
        if 'pairwise_lr_svm' in self.comprehensive_metrics:
            pairwise = self.comprehensive_metrics['pairwise_lr_svm']
            
            effect_size = pairwise['cohens_d_uncertainty']
            
            # Cohen's d interpretation
            if abs(effect_size) < 0.2:
                interpretation = 'Negligible'
                color = 'gray'
            elif abs(effect_size) < 0.5:
                interpretation = 'Small'
                color = 'yellow'
            elif abs(effect_size) < 0.8:
                interpretation = 'Medium'
                color = 'orange'
            else:
                interpretation = 'Large'
                color = 'red'
            
            bar = ax.bar(['Cohen\'s d\n(Uncertainty)'], [abs(effect_size)], color=color, alpha=0.7)
            
            ax.text(0, abs(effect_size) + 0.05, f'{effect_size:.3f}\n({interpretation})', 
                   ha='center', va='bottom', fontsize=12)
            
            ax.set_ylabel('Effect Size |d|', fontsize=12)
            ax.set_title('Effect Size Analysis\n(SVM - LR Uncertainty)', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    def _plot_individual_uncertainty_comparison(self, ax):
        """Plot individual animal uncertainty comparison"""
        lr_uncertainties = self.comprehensive_metrics['Logistic Regression']['uncertainties']
        svm_uncertainties = self.comprehensive_metrics['SVM']['uncertainties']
        
        x = np.arange(len(self.animal_ids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lr_uncertainties, width, label='Logistic Regression', 
                      color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, svm_uncertainties, width, label='SVM', 
                      color='red', alpha=0.7)
        
        ax.set_xlabel('Animal ID', fontsize=12)
        ax.set_ylabel('Classification Uncertainty', fontsize=12)
        ax.set_title('Individual Animal Uncertainty Comparison', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{aid}' for aid in self.animal_ids], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary_table(self, ax):
        """Create performance summary table"""
        # Prepare data for table
        metrics_data = []
        
        for model_name in ['Logistic Regression', 'SVM']:
            if model_name in self.comprehensive_metrics:
                metrics = self.comprehensive_metrics[model_name]
                metrics_data.append([
                    model_name,
                    f"{metrics['cv_accuracy']:.3f}",
                    f"{metrics['mean_uncertainty']:.3f}",
                    f"{metrics['reliability_score']:.3f}",
                    f"{metrics['calibration_consistency']:.3f}",
                    f"{metrics['ccc_score']:.3f}",
                    f"{metrics['cri_score']:.3f}"
                ])
        
        # Add pairwise comparison
        if 'pairwise_lr_svm' in self.comprehensive_metrics:
            pairwise = self.comprehensive_metrics['pairwise_lr_svm']
            metrics_data.append([
                'LR vs SVM',
                f"Δ: {pairwise['uncertainty_wilcoxon_p']:.3f}",
                f"d: {pairwise['cohens_d_uncertainty']:.3f}",
                f"Dom: {pairwise['uncertainty_dominance']}",
                '-',
                '-',
                '-'
            ])
        
        table_headers = ['Model', 'CV Acc', 'Mean U', 'R(h)', 'Cal Cons', 'CCC', 'CRI']
        
        # Create table
        table = ax.table(cellText=metrics_data, colLabels=table_headers,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color coding
        for i in range(len(metrics_data)):
            for j in range(len(table_headers)):
                if i < 2:  # Model rows
                    color = 'lightblue' if i == 0 else 'lightcoral'
                    table[(i+1, j)].set_facecolor(color)
        
        ax.axis('off')
        ax.set_title('Performance Metrics Summary\n(All Theoretical Framework Components)', fontsize=12)
    
    def _plot_feature_importance_comparison(self, ax):
        """Plot feature importance comparison across models"""
        importances_data = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Random Forest
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Logistic Regression, SVM
                importances = np.abs(np.mean(model.coef_, axis=0)) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0])
            else:
                # KNN doesn't have feature importance, skip
                continue
            
            # Get top 10 features for this model
            feature_indices = np.argsort(importances)[-10:][::-1]
            top_features = [self.available_features[i] for i in feature_indices]
            top_importances = importances[feature_indices]
            
            for feat, imp in zip(top_features, top_importances):
                importances_data.append({
                    'Model': model_name,
                    'Feature': feat.replace('_', ' ')[:30],  # Truncate long names
                    'Importance': imp
                })
        
        if importances_data:
            import pandas as pd
            df_imp = pd.DataFrame(importances_data)
            
            # Create grouped bar plot
            models_with_importance = df_imp['Model'].unique()
            x_pos = 0
            colors_map = {'Logistic Regression': 'blue', 'Random Forest': 'green', 'SVM': 'red'}
            
            for model in models_with_importance:
                model_data = df_imp[df_imp['Model'] == model].head(5)  # Top 5 per model
                x_positions = np.arange(x_pos, x_pos + len(model_data))
                
                bars = ax.barh(x_positions, model_data['Importance'], 
                              label=model, alpha=0.7, 
                              color=colors_map.get(model, 'gray'))
                
                # Add feature names
                for i, (pos, feature) in enumerate(zip(x_positions, model_data['Feature'])):
                    ax.text(-0.01, pos, feature, ha='right', va='center', fontsize=8)
                
                x_pos += len(model_data) + 1
            
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title('Top Features by Model\n(Relative Importance Comparison)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Remove y-axis labels since we're adding text
            ax.set_yticks([])
    
    def get_detailed_results(self):
        """Get detailed classification results for all models"""
        print("\n" + "="*60)
        print("DETAILED RESULTS FOR ALL MODELS")
        print("="*60)
        
        all_results = {}
        
        for model_name in self.model_results:
            results = []
            model_result = self.model_results[model_name]
            
            for i in range(len(self.animal_ids)):
                cv_pred = model_result['cv_predictions'][i]
                training_pred = model_result['training_predictions'][i]
                
                # Get probabilities
                if len(model_result['cv_probabilities']) > i:
                    cv_prob = model_result['cv_probabilities'][i]
                else:
                    cv_prob = np.zeros(3)
                    cv_prob[cv_pred] = 1.0
                
                training_prob = model_result['training_probabilities'][i]
                
                results.append({
                    'animal_id': self.animal_ids[i],
                    'true_label': self.label_names[self.labels[i]],
                    'cv_predicted_label': self.label_names[cv_pred],
                    'training_predicted_label': self.label_names[training_pred],
                    'cv_correct': cv_pred == self.labels[i],
                    'training_correct': training_pred == self.labels[i],
                    'cv_confidence': np.max(cv_prob),
                    'training_confidence': np.max(training_prob),
                    'cv_prob_healthy': cv_prob[0],
                    'cv_prob_prediabetic': cv_prob[1],
                    'cv_prob_diabetic': cv_prob[2]
                })
            
            all_results[model_name] = {
                'individual_results': results,
                'training_accuracy': model_result['training_accuracy'],
                'cv_accuracy': model_result['cv_accuracy'],
                'avg_confidence': model_result['avg_confidence']
            }
        
        return all_results
    
    def export_comprehensive_results(self, csv_path):
        """Export comprehensive results for all models"""
        print("\n" + "="*60)
        print("EXPORTING COMPREHENSIVE RESULTS")
        print("="*60)
        
        all_results = self.get_detailed_results()
        
        # Create main results DataFrame
        export_data = []
        
        for i in range(len(self.animal_ids)):
            row = {
                'animal_id': self.animal_ids[i],
                'true_label': self.label_names[self.labels[i]],
                'rat_type_numeric': self.labels[i]
            }
            
            # Add results for each model
            for model_name in all_results:
                result = all_results[model_name]['individual_results'][i]
                prefix = model_name.lower().replace(' ', '_')
                
                row.update({
                    f'{prefix}_cv_prediction': result['cv_predicted_label'],
                    f'{prefix}_training_prediction': result['training_predicted_label'],
                    f'{prefix}_cv_correct': result['cv_correct'],
                    f'{prefix}_training_correct': result['training_correct'],
                    f'{prefix}_cv_confidence': result['cv_confidence'],
                    f'{prefix}_training_confidence': result['training_confidence']
                })
            
            # Add feature values
            for j, feature_name in enumerate(self.available_features):
                if j < self.features_scaled.shape[1]:
                    row[f'feature_{j+1:02d}_{feature_name}_scaled'] = self.features_scaled[i, j]
                    row[f'feature_{j+1:02d}_{feature_name}_raw'] = self.features_raw[i, j]
            
            export_data.append(row)
        
        # Create model summary DataFrame
        model_summary = []
        for model_name in all_results:
            model_summary.append({
                'model_name': model_name,
                'training_accuracy': all_results[model_name]['training_accuracy'],
                'cv_accuracy': all_results[model_name]['cv_accuracy'],
                'avg_confidence': all_results[model_name]['avg_confidence'],
                'overfitting_gap': all_results[model_name]['training_accuracy'] - all_results[model_name]['cv_accuracy']
            })
        
        # Save files
        results_df = pd.DataFrame(export_data)
        summary_df = pd.DataFrame(model_summary)
        
        results_csv = csv_path.replace('.csv', '_multimodel_results.csv')
        summary_csv = csv_path.replace('.csv', '_model_summary.csv')
        
        results_df.to_csv(results_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"[OK] Individual results saved to: {results_csv}")
        print(f"[OK] Model summary saved to: {summary_csv}")
        print(f"[OK] Results shape: {results_df.shape}")
        print(f"[OK] Models compared: {len(summary_df)}")
        
        return results_df, summary_df
    
    def run_complete_analysis(self, csv_path, save_plot_path=None, export_csv=True, 
                             hyperparameter_tuning=True):
        """Run complete multi-model analysis"""
        print("=" * 80)
        print("ENHANCED MULTI-MODEL CLASSIFICATION ANALYSIS")
        print("=" * 80)
        print("Models: Logistic Regression, Random Forest, SVM, KNN")
        print("Same feature selection process maintained")
        print("Comprehensive model comparison")
        print()
        
        try:
            # Step 1: Load and prepare data (identical to original)
            available_features = self.load_and_prepare_data(csv_path)
            
            # Step 2: Train all models
            self.train_all_models(hyperparameter_tuning=hyperparameter_tuning)
            
            # Step 3: Create enhanced visualization
            fig = self.create_enhanced_visualization(save_plot_path)
            
            # Step 3b: Create 15-dimensional analysis visualization
            if save_plot_path:
                path_15d = save_plot_path.replace('.png', '_15d_analysis.png')
            else:
                path_15d = None
            fig_15d = self.create_15d_visualization(path_15d)
            
            # Step 4: Get detailed results
            all_results = self.get_detailed_results()
            
            # Step 5: Export comprehensive results
            if export_csv:
                results_df, summary_df = self.export_comprehensive_results(csv_path)
            else:
                results_df, summary_df = None, None
            
            # Print comprehensive summary
            print("\n" + "=" * 80)
            print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
            print("=" * 80)
            
            print(f"\nPERFORMANCE RANKING (by Cross-Validation Accuracy):")
            sorted_models = sorted(self.model_results.items(), 
                                 key=lambda x: x[1]['cv_accuracy'], reverse=True)
            
            for i, (model_name, results) in enumerate(sorted_models, 1):
                gap = results['training_accuracy'] - results['cv_accuracy']
                print(f"{i}. {model_name:<18}: CV={results['cv_accuracy']:.3f}, "
                      f"Train={results['training_accuracy']:.3f}, "
                      f"Gap={gap:.3f}")
            
            print(f"\n[Best] BEST MODEL: {sorted_models[0][0]} "
                  f"(CV Accuracy: {sorted_models[0][1]['cv_accuracy']:.3f})")
            
            # Overfitting analysis
            print(f"\nOVERFITTING ANALYSIS:")
            print(f"   (Training-CV gap, lower is better)")
            for model_name, results in sorted_models:
                gap = results['training_accuracy'] - results['cv_accuracy']
                status = "Excellent" if gap < 0.1 else "Good" if gap < 0.2 else "Concerning"
                print(f"   • {model_name:<18}: {gap:.3f} ({status})")
            
            return {
                'models': self.models,
                'model_results': self.model_results,
                'all_results': all_results,
                'available_features': available_features,
                'scaler': self.scaler,
                'figure': fig,
                'figure_15d': fig_15d,
                'results_df': results_df,
                'summary_df': summary_df,
                'raw_values': self.features_raw,
                'scaled_values': self.features_scaled,
                'plotted_top2_data': self.plotted_top2_data
            }
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("ENHANCED MULTI-MODEL FED STATE CLASSIFICATION")
    print("=" * 80)
    print("Comparing: Logistic Regression vs Random Forest vs SVM vs KNN")
    print("Same top 15 features from original LR analysis")
    print("Comprehensive model comparison and visualization")
    print()
    
    # Initialize classifier
    classifier = EnhancedMultiModelClassifier()
    
    # Run complete analysis
    try:
        results = classifier.run_complete_analysis(
            csv_path='data/comprehensive_metrics.csv',
            save_plot_path='results/fed_state_multimodel_comparison.png',
            export_csv=True,
            hyperparameter_tuning=True
        )
        
        if results:
            print("\n[SUCCESS] Multi-model analysis completed successfully!")
            print("All four models trained and compared")
            print("Enhanced visualization created")
            print("Comprehensive results exported")
            print("Ready for publication and grant applications")
            
            # Print key insights
            best_model = max(results['model_results'].keys(), 
                           key=lambda x: results['model_results'][x]['cv_accuracy'])
            best_acc = results['model_results'][best_model]['cv_accuracy']
            
            print(f"\nKEY INSIGHTS:")
            print(f"   • Best performing model: {best_model}")
            print(f"   • Best CV accuracy: {best_acc:.3f}")
            print(f"   • All models use identical feature set")
            print(f"   • Comprehensive overfitting analysis included")
            print(f"   • Decision boundary comparison available")
            
        else:
            print("\n[ERROR] Analysis failed - check error messages above")
            
    except FileNotFoundError:
        print("\n[ERROR] Data file not found.")
        print("Please update the csv_path to point to your comprehensive_metrics.csv file")