from mmdet.apis import DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# Perform inference
img_path = 'data/images/test_images/GSAC0346.JPG'
inferencer(img_path, show=True)

models = DetInferencer.list_models('mmdet')