Directory structure:

Download data folder from https://polybox.ethz.ch/index.php/s/HR0yNHW4eCy5Sri

From data.zip, put the data folder inside the code folder extracted from code.zip

The code is implemented in tensorflow 1.13.1 and keras using tensorflow backend (refer requirements.txt for details)

1. data: Contains medical dataset with labels, ALI pretrained weights, GAN pretrained models, ALI z vectors for train dataset
1. CGAN: Contains CGAN code for each dataset. The code for fashion MNIST is obtained from [1] 
1. ACGAN_base: Contains the ACGAN code for each dataset. The code for fashion MNIST is obtained from [2]. 
1. ACGAN_mod_sampling_base:
	1. ACGAN_v1: Discriminator trained with 25% real and 75% fake data in a batch
	1. ACGAN_v2: Discriminator trained with 75% real and 25% fake data in a batch
	1. ACGAN_v2: Discraiminator trained with 75% real and  25% fake data. Every third epoch training with 50% real and 75% fake data.
1. ALI: Contains separate directories for each dataset. Code along the lines of [3]
1. ACGAN_arch2_base: ACGAN with generator matching ALI generator(without using z obtained from ALI). ACGAN_arch2 was run just to compare working with ACGAN_ALI_pretrained but ran into mode collapse. Hence ACGAN_ALI_pretrained was used in all accuracy comparisons.
1. ACGAN_ALI_pretrained: 
	1. V1: Without masking
	1. V2: Masked real/fake loss
1. Classifiers: for imbalanced dataset
	1. Classifier_no_aug
	1. Affine_augmentation
	1. CGAN_augmentation
	1. ACGAN: Contains common classifier for ACGAN_base, ACGAN_mod_sampling
	1. ACGAN_ALI_Classifiers 
1. Image_generation:
	1. cgan_acgan_image_generation.py: File to generate images for CGAN,ACGAN,ACGAN_mod_sampling models
	1. acgan_ali_image_generation.py: File to generate images for ACGAN_ALI_pretrained models
1. Other: Contains previously used classifier based on reduced Efficient-Net tested on complete data. Not included in paper as performance was more or less comparable to previous classifier

Commands:
All the files have to be run in their respective directories
* To run CGAN,ACGAN_base,ACGAN_mod_sampling,ACGAN_arch2_base,ACGAN_ALI_pretrained files:

 ``` python filename.py ```
* To run ALI file: Enter the following command inside the dataset folder

 ``` python train.py ```
* To run Classifier files:
	1. ACGAN : Code for ACGAN_base, ACGAN_mod_sampling models.
	
	 ``` python filename.py --model_name 'model.h5' --num_epoch 100 ```
	
    For example:

    ``` python classifier_fashion_mnist_acgan.py --model_name '../../data/ACGAN_base/acgan_generator_fashion_mnist.h5' --num_epoch 100```
   
    ``` python classifier_fashion_mnist_acgan.py --model_name '../../data/ACGAN_mod_sampling_base/ACGAN_v3/acgan_v3_generator_fashion_mnist.h5' --num_epoch 100```
	1.  ACGAN_ALI_Classifiers:
	For example:

    ```python fashion_mnist.py --model_name '../../data/ACGAN_ALI_pretrained/V1/acgan_ali_v1_generator_fashion_mnist.h5' --num_epoch 100```

	1.  Classifier_no_aug, Affine_augmentation, CGAN_augmentation
 
    ``` python filename.py ```
* To run Image_generation files: arguments are latent_dim, n_classes, model_name, image_name

	* latent_dim=100 for CGAN models, latent_dim=64 for ACGAN models
	* n_classes=10 for fashion MNIST, n_classes=100 for CIFAR100, n_classes=2 for medical dataset
	
        For example:

	``` python cgan_acgan_image_generation.py --model_name '../data/CGAN/cgan_generator_fashion_mnist.h5' --latent_dim 100 --n_classes 10 --image_name 'generated_fashion_mnist.png' ``` 

	``` python acgan_ali_image_generation.py --model_name '../data/ACGAN_ALI_pretrained/V1/acgan_ali_v1_generator_cifar100.h5' --n_classes 100 --image_name 'generated_cifar.png' ```





[1]: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/ 
[2]: https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/ 
[3]: https://github.com/otenim/ALI-Keras2
