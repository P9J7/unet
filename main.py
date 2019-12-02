from model import *
from data import *
import os
import keras as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# data_gen_args = dict(rotation_range=0.2,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
myGene = overlapTrainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = None, target_size=(388, 388))

model = rawUnet()
print(model.summary())
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=100, epochs=10)


# intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[9].output])
# intermediate_tensor = intermediate_tensor_function([])[0]

# testGene = testGenerator("data/membrane/test")
testGene = overlapTestGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/dd", results)
