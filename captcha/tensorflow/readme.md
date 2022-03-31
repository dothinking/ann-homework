# Play CAPTCHA with TensorFlow

Solving captcha using CNN model with multi-labels. Two kinds of datasets are used:

- captcha generated with Python module `captcha`
- qq captcha: true captcha with labels recognized manually

## Folder Structure

```
captcha
	|- samples           captcha images
		|- example
		|- captcha
		|- qq
	|- dataset           TFRecord files converted from samples
	|- models            saved CNN models
	|- tensorboard       tensorboard visulization
```

## Contents

- explore data
	- QQ captcha
	- python captcha
- prepare dataset
	- QQ captcha
	- python captcha
- cnn model with multi-outputs
- cnn model with cropped inputs