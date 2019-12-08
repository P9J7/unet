from model import *
from data import *
import os
import matplotlib.pyplot as plt
import keras as K
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
data_gen_args1 = dict(rotation_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args)
myGene2 = trainGenerator(2, 'data/membrane/val', 'image', 'label', data_gen_args1)
myGene1 = overlapTrainGenerator(1, 'data/membrane/train', 'image', 'label', data_gen_args, target_size=(388, 388))
myGene3 = overlapTrainGenerator(1, 'data/membrane/val', 'image', 'label', data_gen_args1, target_size=(388, 388))

model = unet()
model1 = rawUnet()
# print(model.summary())
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history1 = model1.fit_generator(myGene1, steps_per_epoch=100, epochs=15, validation_data= myGene3, validation_steps=10)
history = model.fit_generator(myGene, steps_per_epoch=100, epochs=15, validation_data= myGene2, validation_steps=10)


# intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[9].output])
# intermediate_tensor = intermediate_tensor_function([])[0]

# testGene = testGenerator("data/membrane/test")
# testGene = overlapTestGenerator("data/membrane/test")
# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("data/membrane/dd", results)
epochs=range(len(history.history['acc']))
epochs1=range(len(history1.history['acc']))

plt.figure()
plt.plot(epochs,history.history['acc'],'b',label='OpenSource U-Net Training acc')
plt.plot(epochs,history.history['val_acc'],'r',label='OpenSource U-Net Validation acc')
plt.plot(epochs1,history1.history['acc'],'g',label='Implemented U-Net Training acc')
plt.plot(epochs1,history1.history['val_acc'],'y',label='Implemented U-Net Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.savefig('rawUnet_unet_acc.jpg')

plt.figure()
plt.plot(epochs, history.history['loss'],'b',label='OpenSource U-Net Training loss')
plt.plot(epochs, history.history['val_loss'],'r',label='OpenSource U-Net Validation val_loss')
plt.plot(epochs1, history1.history['loss'],'g',label='Implemented U-Net Training loss')
plt.plot(epochs1, history1.history['val_loss'],'y',label='Implemented U-Net Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('rawUnet_unet_loss.jpg')
