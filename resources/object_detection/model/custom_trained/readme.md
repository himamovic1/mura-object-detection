# How to export a model (workaround)

execute:
```
python -m scripts.tf_model_export_workaround --input_type image_tensor --pipeline_config_path .\resources\object_detection\model\custom\pipeline.config --trained_checkpoint_dir .\resources\object_detection\model\custom --output_directory .\resources\object_detection\model\custom_trained
```