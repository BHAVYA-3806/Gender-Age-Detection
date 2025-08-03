# Gender and Age Detection using Deep Learning

## 🎯 Objective
The goal of this project is to build a deep learning-based system that can predict the gender and estimate the age group of a person from an image or a webcam feed.

---

## 📌 Project Overview
In this Python-based project, a deep learning approach is used to recognize the gender and age group of a person from a facial image. The system is built using pre-trained models by Tal Hassner and Gil Levi. Gender prediction is binary: either 'Male' or 'Female', while age prediction falls into one of eight predefined categories:

- (0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100)

Because estimating an exact age from a single image is highly sensitive to variables like lighting, expression, and makeup, the problem is framed as a classification task rather than regression.

---

## 📂 Dataset Used
The model is trained using the **Adience Dataset**, which is publicly available and contains face images under real-world conditions (like poor lighting, varied poses, and occlusions). It consists of:

- 🧍‍♂️ 26,580 images  
- 👥 2,284 unique subjects  
- 📁 1GB of total data  

The dataset was compiled from Flickr albums and is shared under a Creative Commons license.

---

## 📦 Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install opencv-python
pip install argparse
```
---

## ⚙️ How to Use

1. **Clone or download** this repository to your local machine.

2. **Open Command Prompt or Terminal** and navigate to the project folder.

### 🖼️ To Detect Age and Gender from an Image

```bash
python detect.py --image <image_name>
```
✅ Make sure the image is placed in the same folder as the script.

### 🎥 To Use Webcam for Live Detection

```bash
python detect.py
```
Press Ctrl + C to stop the webcam feed.

---

### 📸 Examples

```bash
> python detect.py --image girl1.jpg
Gender: Female
Age: 25–32 years

![Girl1](output_girl1.png)

> python detect.py --image kid1.jpg
Gender: Male
Age: 4–6 years

![Kid1](output_kid1.png)

> python detect.py --image man1.jpg
Gender: Male
Age: 38–43 years

![Man1](output_man1.png)

> python detect.py --image woman1.jpg
Gender: Female
Age: 38–43 years

![Woman1](output_woman1.png)
```
---

## 📌 Disclaimer

Images used in the sample outputs are sourced from public domains and are used strictly for educational purposes.  
If there are any concerns, they will be promptly addressed and removed.

---

## 🔗 Credits

- Pre-trained models by **Tal Hassner** and **Gil Levi**
---


