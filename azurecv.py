# Azure Custom Vision
# Serge Retkowsky - Microsoft - serge.retkowsky@microsoft.com
# 15/09/2025

import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from collections import Counter
from dotenv import load_dotenv
from IPython.display import display
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, classification_report


class AzureCustomVisionImageClassifier:
    def __init__(self, training_endpoint, prediction_endpoint, training_key, prediction_key, prediction_resource_id):
        """Initialize the Custom Vision client"""
        self.endpoint = training_endpoint
        self.endpoint2 = prediction_endpoint
        self.training_key = training_key
        self.prediction_key = prediction_key
        self.prediction_resource_id = prediction_resource_id
        
        # Create training and prediction clients
        training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        
        self.trainer = CustomVisionTrainingClient(training_endpoint, training_credentials)
        self.predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)
        self.project = None
        self.published_model_name = None

    def create_project(self, project_name, domain_id, classification_type, description="Image classification project"):
        """Create a new Custom Vision project"""
        print(f"🛠️ Creating project: {project_name}")
        
        # Get available domains to find General [A2]
        domains = self.trainer.get_domains()

        self.project = self.trainer.create_project(
          project_name, 
          description,
          domain_id=domain_id,
          classification_type=classification_type
        )

        print(f"\n✅ Project created with ID: {self.project.id}")
        return self.project

    def list_available_domains(self):
        """List all available domains for Custom Vision projects"""
        print("Available Azure Custom Vision domains:")
        print("=" * 100)
        
        try:
            domains = self.trainer.get_domains()

            # Group domains by type
            domain_groups = {}
            for domain in domains:
                domain_type = domain.type if hasattr(domain, 'type') else 'Unknown'
                if domain_type not in domain_groups:
                    domain_groups[domain_type] = []
                domain_groups[domain_type].append(domain)
            
            for domain_type, domain_list in domain_groups.items():
                print(f"\n{domain_type.upper()} DOMAINS:\n")
                for domain in domain_list:
                    exportable = "✅ Exportable" if getattr(domain, 'exportable', False) else "ℹ️ Cloud only"
                    print(f"  • {domain.name:<25} {exportable}\tID: {domain.id}")
                    
        except Exception as e:
            print(f"Error retrieving domains: {e}")

    def get_existing_project(self, project_name):
        """Get an existing project by name"""
        projects = self.trainer.get_projects()

        for project in projects:
            if project.name == project_name:
                self.project = project
                print(f"Found existing project: {project_name}")
                return project

        print(f"Project '{project_name}' not found")
        return None

    def create_tags(self, tag_names):
        """Create tags for the project"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        tags = {}
        for tag_name in tag_names:
            tag = self.trainer.create_tag(self.project.id, tag_name)
            tags[tag_name] = tag
            print(f"✅ Created tag: {tag_name}")

        return tags

    def upload_images_from_folder(self, folder_path, tag_name, tags_dict, max_images_per_batch=64):
        """Upload images from a folder with a specific tag"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        if tag_name not in tags_dict:
            raise ValueError(f"Tag '{tag_name}' not found in tags dictionary")

        tag = tags_dict[tag_name]
        image_files = []

        # Get all image files from folder
        supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(folder_path, filename)
                with open(image_path, "rb") as image_contents:
                    image_files.append(ImageFileCreateEntry(
                        name=filename,
                        contents=image_contents.read(),
                        tag_ids=[tag.id]
                    ))

        if not image_files:
            print(f"❌ No images found in folder: {folder_path}")
            return

        # Upload images in batches
        total_uploaded = 0
        for i in range(0, len(image_files), max_images_per_batch):
            batch = image_files[i:i + max_images_per_batch]
            upload_result = self.trainer.create_images_from_files(
                self.project.id,
                ImageFileCreateBatch(images=batch)
            )

            if upload_result.is_batch_successful:
                total_uploaded += len(batch)
                print(f"✅ Uploaded batch of {len(batch)} images for tag '{tag_name}'")
            else:
                print(f"❌ Failed to upload batch. Errors: {upload_result.images}")

        print(f"Total images uploaded for '{tag_name}' = {total_uploaded}\n")

    def upload_single_image(self, image_path, tag_name, tags_dict):
        """Upload a single image with a tag"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        if tag_name not in tags_dict:
            raise ValueError(f"Tag '{tag_name}' not found in tags dictionary")

        tag = tags_dict[tag_name]

        with open(image_path, "rb") as image_contents:
            upload_result = self.trainer.create_images_from_files(
                self.project.id,
                ImageFileCreateBatch(images=[
                    ImageFileCreateEntry(
                        name=os.path.basename(image_path),
                        contents=image_contents.read(),
                        tag_ids=[tag.id]
                    )
                ])
            )

        if upload_result.is_batch_successful:
            print(f"✅ Successfully uploaded: {image_path}")
        else:
            print(f"❌ Failed to upload: {image_path}")

    def train_model_regular(self, iteration_name=None, timemsg=10):
        """Train the model with uploaded images"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        print("🧠 Starting regular training...\n")

        if iteration_name is None:
            iteration_name = f"Iteration_{int(time.time())}"

        iteration = self.trainer.train_project(self.project.id)
        
        # Wait for training to complete
        while iteration.status != "Completed":
            iteration = self.trainer.get_iteration(self.project.id, iteration.id)
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d %H:%M:%S")  # Fixed: removed extra parenthesis
            print(f"⏳ {dt} Training status: {iteration.status}")
            time.sleep(timemsg)

        print("\n✅ Training completed!")
        print(f"Iteration ID: {iteration.id}")

        return iteration

    def train_model_advanced(self, iteration_name=None, timemsg=60, trainhoursmax=3, email=None):
        """Train the model with uploaded images"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        print("🧠 Starting advanced training...\n")

        iteration = self.trainer.train_project(
            self.project.id,
            training_type="Advanced",
            reserved_budget_in_hours=trainhoursmax,
            forceTrain = True,
            notification_email_address=email,
        )
        
        # Wait for training to complete
        while iteration.status != "Completed":
            iteration = self.trainer.get_iteration(self.project.id, iteration.id)
            now = datetime.datetime.now()
            dt = now.strftime("%Y-%m-%d %H:%M:%S")  # Fixed: removed extra parenthesis
            print(f"⏳ {dt} Training status: {iteration.status}")
            time.sleep(timemsg)

        print("\n✅ Training completed!")
        print(f"Iteration ID: {iteration.id}")

        return iteration
          
    def publish_model(self, iteration, model_name, prediction_resource_id=None):
        """Publish the trained model for prediction"""
        if not self.project:
            raise ValueError("No project selected. Create or get a project first.")

        if prediction_resource_id is None:
            prediction_resource_id = self.prediction_resource_id

        self.published_model_name = model_name

        # Publish the iteration
        publish_iteration_name = model_name
        
        self.trainer.publish_iteration(
            self.project.id,
            iteration.id,
            publish_iteration_name,
            prediction_resource_id
        )

        print(f"✅ Model published as: {model_name}")

    def predict_image_from_file(self, image_path):
        """Make prediction on an image file"""
        if not self.project or not self.published_model_name:
            raise ValueError("Project not set up or model not published")

        try:
            with open(image_path, "rb") as image_contents:
                results = self.predictor.classify_image(
                    self.project.id,
                    self.published_model_name,
                    image_contents.read()
                )

            predictions = []

            for prediction in results.predictions:
                predictions.append({
                    'tag': prediction.tag_name,
                    'probability': prediction.probability
                })

            sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
            
            return sorted_predictions

        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg:
                print("❌ AUTHENTICATION ERROR:")
                print("1. Check your PREDICTION_KEY is correct")
                print("2. Verify your ENDPOINT URL is correct")
                print("3. Ensure your model is published")
                print("4. Check if your Azure subscription is active")
                self._debug_prediction_setup()
            elif "not found" in error_msg:
                print("❌ MODEL NOT FOUND ERROR:")
                print("Make sure your model is published with the correct name")
                self._debug_prediction_setup()
            else:
                print(f"❌ PREDICTION ERROR: {e}")

            return []

    def predict_image_from_url(self, image_url):
        """Make prediction on an image from URL"""
        if not self.project or not self.published_model_name:
            raise ValueError("Project not set up or model not published")

        try:
            results = self.predictor.classify_image_url(
                self.project.id,
                self.published_model_name,
                url=image_url
            )

            predictions = []

            for prediction in results.predictions:
                predictions.append({
                    'tag': prediction.tag_name,
                    'probability': prediction.probability
                })

            sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
            
            return sorted_predictions

        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg:
                print("❌ AUTHENTICATION ERROR:")
                print("1. Check your PREDICTION_KEY is correct")
                print("2. Verify your ENDPOINT URL is correct")
                print("3. Ensure your model is published")
                self._debug_prediction_setup()
            else:
                print(f"❌ PREDICTION ERROR: {e}")

            return []

    def get_project_info(self):
        """Get information about the current project"""
        if not self.project:
            raise ValueError("No project selected.")
        
        # Get project details
        project_details = self.trainer.get_project(self.project.id)
        
        # Get tags
        tags = self.trainer.get_tags(self.project.id)
        
        # Get iterations
        iterations = self.trainer.get_iterations(self.project.id)
        
        print(f"\n🧠 Azure Custom Vision project: {project_details.name}")
        print(f"📝 Description: {project_details.description}")
        print(f"🧾 Project ID: {project_details.id}")
        
        print(f"\n🏷️ Tags ({len(tags)}):")
        for tag in tags:
            print(f"  - {tag.name} (ID: {tag.id})")
        
        print(f"\n🔁 Iterations ({len(iterations)}):")
        
        for iteration in iterations:
            status = "Published" if iteration.publish_name else "Not Published"
            print(f"  - 📰 {iteration.name}: {iteration.status} ({status})")
    
    def get_model_performance_metrics(self, iteration_id=None, threshold=0.5):
        """
        Get detailed performance metrics for a trained model including mAP, precision, and recall
        
        Args:
            iteration_id: Specific iteration ID. If None, uses the latest iteration
            threshold: Probability threshold for predictions (default: 0.5)
        
        Returns:
            dict: Performance metrics including overall and per-tag metrics
        """
        if not self.project:
            raise ValueError("No project selected.")
        
        # Get the iteration to evaluate
        if iteration_id is None:
            iterations = self.trainer.get_iterations(self.project.id)
            if not iterations:
                raise ValueError("No iterations found for this project")
            # Get the latest completed iteration
            completed_iterations = [it for it in iterations if it.status == "Completed"]
            if not completed_iterations:
                raise ValueError("No completed iterations found")
            iteration = completed_iterations[-1]
        else:
            iteration = self.trainer.get_iteration(self.project.id, iteration_id)
        
        print(f"\nGetting performance metrics for iteration: {iteration.name}")
        print(f"Iteration ID: {iteration.id}")
        print(f"Training completed: {iteration.created}")
        
        # Get iteration performance
        performance = self.trainer.get_iteration_performance(self.project.id, iteration.id, threshold)
        
        # Get tags for reference
        tags = self.trainer.get_tags(self.project.id)
        tag_dict = {tag.id: tag.name for tag in tags}
        
        # Compile performance metrics
        metrics = {
            'iteration_info': {
                'id': iteration.id,
                'name': iteration.name,
                'status': iteration.status,
                'created': str(iteration.created),
                'threshold': threshold
            },
            'overall_metrics': {
                'precision': performance.precision,
                'recall': performance.recall,
                'average_precision': performance.average_precision  # This is the mAP
            },
            'per_tag_metrics': []
        }
        
        # Add per-tag performance metrics
        if hasattr(performance, 'per_tag_performance') and performance.per_tag_performance:
            for tag_perf in performance.per_tag_performance:
                tag_name = tag_dict.get(tag_perf.id, f"Tag_{tag_perf.id}")
                tag_metrics = {
                    'tag_name': tag_name,
                    'tag_id': tag_perf.id,
                    'precision': tag_perf.precision,
                    'recall': tag_perf.recall,
                    'average_precision': tag_perf.average_precision
                }
                metrics['per_tag_metrics'].append(tag_metrics)
        
        # Print formatted results
        self._print_performance_metrics(metrics)
        
        return metrics
    
    def _print_performance_metrics(self, metrics):
        """Print formatted performance metrics"""
        print("\n" + "="*100)
        print("🚀 Model Performance Metrics")
        print("="*100)
        
        # Overall metrics
        overall = metrics['overall_metrics']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"├── Precision =           {overall['precision']:.5f} ({overall['precision']*100:.2f}%)")
        print(f"├── Recall =              {overall['recall']:.5f} ({overall['recall']*100:.2f}%)")
        print(f"└── mAP (Avg Precision) = {overall['average_precision']:.5f} ({overall['average_precision']*100:.2f}%)")
        
        # F1 Score calculation
        precision = overall['precision']
        recall = overall['recall']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    F1-Score =            {f1_score:.4f} ({f1_score*100:.2f}%)")
        
        # Per-tag metrics
        if metrics['per_tag_metrics']:
            print(f"\nPER-TAG PERFORMANCE:")
            print("-" * 60)
            print(f"{'Tag Name':<15} {'Precision':<12} {'Recall':<12} {'mAP':<12} {'F1-Score':<12}")
            print("-" * 60)
            
            for tag_metrics in metrics['per_tag_metrics']:
                tag_precision = tag_metrics['precision']
                tag_recall = tag_metrics['recall']
                tag_map = tag_metrics['average_precision']
                tag_f1 = 2 * (tag_precision * tag_recall) / (tag_precision + tag_recall) if (tag_precision + tag_recall) > 0 else 0
                
                print(f"{tag_metrics['tag_name']:<15} "
                      f"{tag_precision:.5f}      "
                      f"{tag_recall:.5f}      "
                      f"{tag_map:.5f}      "
                      f"{tag_f1:.5f}")
        
        print("\n" + "="*100)
    
    def compare_model_iterations(self, threshold=0.5):
        """
        Compare performance metrics across all completed iterations
        
        Args:
            threshold: Probability threshold for predictions
            
        Returns:
            list: List of performance metrics for each iteration
        """
        if not self.project:
            raise ValueError("No project selected.")
        
        iterations = self.trainer.get_iterations(self.project.id)
        completed_iterations = [it for it in iterations if it.status == "Completed"]
        
        if not completed_iterations:
            print("❌ No completed iterations found.")
            return []
        
        print(f"\nComparing {len(completed_iterations)} completed iterations:")
        print("="*80)
        print(f"{'Iteration':<20} {'Precision':<12} {'Recall':<12} {'mAP':<12} {'F1-Score':<12} {'Date':<20}")
        print("="*80)
        
        all_metrics = []
        
        for iteration in completed_iterations:
            try:
                performance = self.trainer.get_iteration_performance(self.project.id, iteration.id, threshold)
                
                precision = performance.precision
                recall = performance.recall
                map_score = performance.average_precision
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                iteration_metrics = {
                    'iteration_name': iteration.name,
                    'iteration_id': iteration.id,
                    'precision': precision,
                    'recall': recall,
                    'map': map_score,
                    'f1_score': f1_score,
                    'created': iteration.created
                }
                
                all_metrics.append(iteration_metrics)
                
                print(f"{iteration.name:<20} "
                      f"{precision:.4f}      "
                      f"{recall:.4f}      "
                      f"{map_score:.4f}      "
                      f"{f1_score:.4f}      "
                      f"{str(iteration.created)[:19]}")
                
            except Exception as e:
                print(f"❌ Error getting metrics for {iteration.name}: {str(e)}")
        
        # Find best performing iteration
        if all_metrics:
            best_iteration = max(all_metrics, key=lambda x: x['f1_score'])
            print("\n" + "="*80)
            print(f"🏅BEST PERFORMING ITERATION (by F1-Score): {best_iteration['iteration_name']}")
            print(f"F1-Score: {best_iteration['f1_score']:.4f}, Precision: {best_iteration['precision']:.4f}, "
                  f"Recall: {best_iteration['recall']:.4f}, mAP: {best_iteration['map']:.4f}")
        
        return all_metrics

    def _debug_prediction_setup(self):
        """Helper method to debug prediction setup issues"""
        print("\n🔍 DEBUG INFO:")
        print(f"  Endpoint: {self.prediction_endpoint}")
        print(f"  Project ID: {self.project.id if self.project else 'None'}")
        print(f"  Published Model Name: {self.published_model_name}")
        print(f"  Prediction Resource ID: {self.prediction_resource_id}")

    def get_detailed_model_evaluation(self, iteration_id=None, export_to_file=None):
        """
        Get comprehensive model evaluation including confusion matrix data
        
        Args:
            iteration_id: Specific iteration ID. If None, uses the latest iteration
            export_to_file: Optional file path to export metrics as JSON
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if not self.project:
            raise ValueError("No project selected.")
        
        # Get performance metrics
        metrics = self.get_model_performance_metrics(iteration_id)
        
        # Get additional project information
        tags = self.trainer.get_tags(self.project.id)
        
        # Add tag information
        metrics['tag_information'] = []
        for tag in tags:
            tag_info = {
                'name': tag.name,
                'id': tag.id,
                'description': tag.description if hasattr(tag, 'description') else None,
                'image_count': tag.image_count if hasattr(tag, 'image_count') else 'Unknown'
            }
            metrics['tag_information'].append(tag_info)
        
        # Export to file if requested
        if export_to_file:
            with open(export_to_file, 'w') as f:
                # Convert any datetime objects to strings for JSON serialization
                json_metrics = self._prepare_for_json(metrics)
                json.dump(json_metrics, f, indent=2)
        
        print(f"\nMetrics exported to: {export_to_file}")
        
        return metrics
    
    def _prepare_for_json(self, obj):

        from datetime import datetime
        
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def generate_confusion_matrix(self, test_images_folder, iteration_id=None, threshold=0.5, 
                                 save_plot=True, plot_filename="confusion_matrix.png"):
        """
        Generate confusion matrix by testing the model on a set of labeled test images
        
        Args:
            test_images_folder: Dictionary mapping tag names to folders containing test images
                               Example: {"cat": "path/to/cat/test", "dog": "path/to/dog/test"}
            iteration_id: Specific iteration ID. If None, uses the published model
            threshold: Probability threshold for predictions
            save_plot: Whether to save the confusion matrix plot
            plot_filename: Filename for saved plot
            
        Returns:
            dict: Confusion matrix data and metrics
        """        
        if not self.project or not self.published_model_name:
            raise ValueError("Project not set up or model not published")
        
        # Get tags
        tags = self.trainer.get_tags(self.project.id)
        tag_names = [tag.name for tag in tags]
        tag_name_to_id = {tag.name: tag.id for tag in tags}
        
        print(f"Generating confusion matrix for {len(tag_names)} classes: {tag_names}")
        
        # Collect predictions and true labels
        true_labels = []
        predicted_labels = []
        prediction_probabilities = []
        image_paths = []
        
        # Process each test folder
        for true_tag, folder_path in test_images_folder.items():
            if true_tag not in tag_names:
                print(f"❌ Warning: Tag '{true_tag}' not found in model tags. Skipping.")
                continue
                
            if not os.path.exists(folder_path):
                print(f"❌ Warning: Folder '{folder_path}' not found. Skipping.")
                continue
            
            # Get all image files
            supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(supported_formats)]
            
            print(f"🔄 Processing {len(image_files)} images from {true_tag} folder...")
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                
                try:
                    # Make prediction
                    with open(image_path, "rb") as image_contents:
                        results = self.predictor.classify_image(
                            self.project.id, 
                            self.published_model_name, 
                            image_contents.read()
                        )
                    
                    # Get the prediction with highest probability
                    if results.predictions:
                        best_prediction = max(results.predictions, key=lambda x: x.probability)
                        predicted_tag = best_prediction.tag_name
                        max_probability = best_prediction.probability
                        
                        # Only consider predictions above threshold
                        if max_probability >= threshold:
                            predicted_labels.append(predicted_tag)
                        else:
                            predicted_labels.append("Unknown")  # Below threshold
                    else:
                        predicted_labels.append("Unknown")
                        max_probability = 0.0
                    
                    true_labels.append(true_tag)
                    prediction_probabilities.append(max_probability)
                    image_paths.append(image_path)

                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {str(e)}")
                    continue
        
        if not true_labels:
            raise ValueError("No valid predictions were made. Check your test data and model.")
        
        print(f"🛠️ Generated predictions for {len(true_labels)} images")
        
        # Create confusion matrix
        all_labels = sorted(list(set(true_labels + predicted_labels)))
        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
        
        # Calculate metrics
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        
        # Create detailed results
        confusion_data = {
            'confusion_matrix': cm.tolist(),
            'labels': all_labels,
            'accuracy': accuracy,
            'total_predictions': len(true_labels),
            'threshold_used': threshold,
            'detailed_results': []
        }
        
        # Add detailed per-image results
        for i, (true_label, pred_label, prob, img_path) in enumerate(
            zip(true_labels, predicted_labels, prediction_probabilities, image_paths)):
            confusion_data['detailed_results'].append({
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': pred_label,
                'probability': prob,
                'correct': true_label == pred_label
            })
        
        # Print confusion matrix
        self._print_confusion_matrix(cm, all_labels, accuracy)
        
        # Generate classification report
        try:

            report = classification_report(true_labels, predicted_labels, zero_division=0,
                                         labels=all_labels, output_dict=True)
            confusion_data['classification_report'] = report
            self._print_classification_report(report)
        except ImportError:
            print("❌ sklearn not available for detailed classification report")
        
        # Plot confusion matrix
        if save_plot:
            self._plot_confusion_matrix(cm, all_labels, accuracy, plot_filename)
        
        return confusion_data
    
    def _print_confusion_matrix(self, cm, labels, accuracy):
        """Print formatted confusion matrix"""
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Test Images: {np.sum(cm)}")
        
        # Print matrix header
        #print(f"\n{'True \\ Predicted':<15}", end="")
        for label in labels:
            print(f"{label:>10}", end="")
        print(" | Total")
        print("-" * (15 + 10*len(labels) + 8))
        
        # Print matrix rows
        for i, true_label in enumerate(labels):
            print(f"{true_label:<15}", end="")
            row_total = np.sum(cm[i, :])
            for j, pred_label in enumerate(labels):
                print(f"{cm[i, j]:>10}", end="")
            print(f" | {row_total}")
        
        # Print column totals
        print("-" * (15 + 10*len(labels) + 8))
        print(f"{'Total':<15}", end="")
        for j in range(len(labels)):
            col_total = np.sum(cm[:, j])
            print(f"{col_total:>10}", end="")
        print(f" | {np.sum(cm)}")
        
        # Print per-class accuracy
        print(f"\nPER-CLASS ACCURACY:")
        for i, label in enumerate(labels):
            if np.sum(cm[i, :]) > 0:
                class_accuracy = cm[i, i] / np.sum(cm[i, :])
                print(f"  {label}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    def _print_classification_report(self, report):
        """Print sklearn classification report"""
        print("\n" + "="*60)
        print("🧠 DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                print(f"{class_name:<15} "
                      f"{metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} "
                      f"{int(metrics['support']):<10}")
        
        print("-" * 60)
        if 'weighted avg' in report:
            avg = report['weighted avg']
            print(f"{'Weighted Avg':<15} "
                  f"{avg['precision']:<10.4f} "
                  f"{avg['recall']:<10.4f} "
                  f"{avg['f1-score']:<10.4f} "
                  f"{int(avg['support']):<10}")
    
    def _plot_confusion_matrix(self, cm, labels, accuracy, filename):
        """Create and save confusion matrix visualization"""
        try:            
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Number of Predictions'})
            
            plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nConfusion matrix plot saved as: {filename}")
            
        except ImportError as e:
            print(f"❌ Could not create plot. Missing library: {e}")
        except Exception as e:
            print(f"❌ Error creating confusion matrix plot: {e}")
    
    def analyze_misclassifications(self, confusion_data, top_n=5):
        """
        Analyze the most common misclassifications
        
        Args:
            confusion_data: Output from generate_confusion_matrix()
            top_n: Number of top misclassifications to show
            
        Returns:
            list: Most common misclassification patterns
        """
        if 'detailed_results' not in confusion_data:
            print("❌ No detailed results available for misclassification analysis")
            return []
        
        # Find all misclassifications
        misclassifications = []
        for result in confusion_data['detailed_results']:
            if not result['correct'] and result['predicted_label'] != 'Unknown':
                misclassifications.append({
                    'true_label': result['true_label'],
                    'predicted_label': result['predicted_label'],
                    'probability': result['probability'],
                    'image_path': result['image_path']
                })
        
        if not misclassifications:
            print("❌ No misclassifications found!")
            return []
        
        # Count misclassification patterns
        patterns = Counter()
        for misc in misclassifications:
            pattern = f"{misc['true_label']} → {misc['predicted_label']}"
            patterns[pattern] += 1
        
        print(f"\n" + "="*60)
        print("📊 MISCLASSIFICATION ANALYSIS")
        print("="*60)
        print(f"⚠️ Total misclassifications: {len(misclassifications)}")
        print(f"✅ Total correct predictions: {len(confusion_data['detailed_results']) - len(misclassifications)}")
        
        print(f"\n⚠️ TOP {top_n} MISCLASSIFICATION PATTERNS:")
        print("-" * 40)
        
        most_common = patterns.most_common(top_n)
        for i, (pattern, count) in enumerate(most_common, 1):
            percentage = (count / len(confusion_data['detailed_results'])) * 100
            print(f"{i}. {pattern}: {count} times ({percentage:.2f}%)")
        
        if most_common:
            top_pattern = most_common[0][0]
            true_label, pred_label = top_pattern.split(' → ')
            
            print(f"\nEXAMPLE IMAGES FOR '{top_pattern}':")
            examples = [m for m in misclassifications 
                       if m['true_label'] == true_label and m['predicted_label'] == pred_label]
            
            for i, example in enumerate(examples):
                print(f"  {i+1}. {os.path.basename(example['image_path'])} "
                      f"(confidence: {example['probability']:.3f})")
        
        return most_common

