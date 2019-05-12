from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
trainGene = trainGenerator(1,'data/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print("before fit generator")
model.fit_generator(trainGene,steps_per_epoch=500,epochs=2,callbacks=[model_checkpoint])
print("after fit generator")

testGene = testGenerator("data/test", 3)
results = model.predict_generator(testGene, 3,verbose=1)
print("after predict generator")
saveResult("data/test",results)