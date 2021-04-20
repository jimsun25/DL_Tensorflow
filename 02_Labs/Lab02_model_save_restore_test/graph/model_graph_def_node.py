import tensorflow as tf
import os

model_dir = './pb_model'
model_name = 'classify_image_graph_def.pb'

def create_graph():
  with tf.gfile.FastGFile(os.path.join(model_dir, model_name), 'rb') as f:
  
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # Imports the graph from graph_def into the current default Graph.
    tf.import_graph_def(graph_def, name='')
    
create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
  print(tensor_name,'\n')







