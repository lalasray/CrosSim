from Pretraining_JLR_11 import train_model
h5_file_path = "../../../../ds/other/CrosSim/UniMocap/full_dataset.h5"
train_model(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_JLR_ft_11 import train_model_ft
train_model_ft(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_bimodal_textimu_ft_11 import train_bimodel_ft
train_bimodel_ft(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_bimodal_textimu_11 import train_bimodel
train_bimodel(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_bimodal_imupose_11 import train_bipose
train_bipose(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_JLR_imubi_11 import train_model_imubi
train_model_imubi(epochs=300, batch_size=512,h5_file_path = h5_file_path)

from Pretraining_JLR_imubi_ft_11 import train_model_imubi_ft
train_model_imubi_ft(epochs=300, batch_size=512,h5_file_path = h5_file_path)