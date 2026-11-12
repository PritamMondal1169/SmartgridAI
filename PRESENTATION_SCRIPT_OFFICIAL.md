# SmartGrid AI — Official 10-Member Presentation Script
**Format:** 60 Seconds per Speaker | Exactly 10 Minutes Total | Structured for Slides 1 to 15  

---

## Speaker Index & Slide Mapping

| # | Speaker Name | Role / Topic | Slides |
|:---:|:---|:---|:---:|
| **01** | **Sohan Ahmed (Leader)** | Project and Team Introduction | Slides 1–2 |
| **02** | **Angsu Dolai** | Problem Statement | Slides 3–4 |
| **03** | **Arunabha Maiti** | Proposed Solution and Objectives | Slides 5–6 |
| **04** | **Binita Roy** | Datasets and System Architecture | Slides 7–8 |
| **05** | **Dyutimoy Panda** | Household Electricity Prediction (ANN) | Slide 9 |
| **06** | **Enakhya Mandal** | Deep Neural Network Training (DNN) | Slide 10 |
| **07** | **Pritam** | AI-Based Infrastructure Inspection (Computer Vision) | Slide 11 |
| **08** | **Rittick Raj Mishra** | Electricity Demand Forecasting (Sequence Models) | Slide 12 |
| **09** | **Sahil Ahmed** | Synthetic Data & Anomaly Detection (GAN & VAE) | Slide 13 |
| **10** | **Supriyo Jana** | Expected Results, Applications & Conclusion | Slides 14–15 |

---

## 01 Sohan Ahmed (Leader)
**Topic:** Project and Team Introduction  
**Slide Coverage:** Slides 1–2  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"Good morning, everyone. I'm Sohan Ahmed, the team leader. Our project is called **SmartGrid AI**. We're looking at how electricity readings and photos of power equipment can help people make better decisions about the power grid."*

* **20–40 sec:**  
  > *"Here are the ten members of our team. During this presentation, we'll show how the project could estimate future demand, highlight unusual electricity use, and help inspection teams review photos of equipment such as utility poles."*

* **40–60 sec:**  
  > *"Our idea is to put these results together in one dashboard so users can see what needs their attention. I'll now hand over to **Angsu Dolai**, who will explain the problems that led us to this project."*

---

## 02 Angsu Dolai
**Topic:** Problem Statement  
**Slide Coverage:** Slides 3–4  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"Electricity use changes throughout the day. A daily average can hide the hours when people need the most power. If we can estimate when demand will peak, the people who run the grid can plan ahead."*

* **20–40 sec:**  
  > *"It's also hard to tell whether a high meter reading is unusual without knowing what normal usage looks like. For instance, a rise around dinner time might be routine for one home and unusual for another."*

* **40–60 sec:**  
  > *"Equipment inspections create a different challenge. Staff may need to review many field photos to find the ones that deserve attention. SmartGrid AI brings these problems together. **Arunabha Maiti** will now explain our proposed solution."*

---

## 03 Arunabha Maiti
**Topic:** Proposed Solution and Objectives  
**Slide Coverage:** Slides 5–6  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"Our proposal is to bring five AI modules into one dashboard. The first estimates household electricity use, and the second compares different ways of training a neural network. The third uses field photos to support equipment inspections."*

* **20–40 sec:**  
  > *"The fourth module forecasts electricity demand using past readings. The fifth creates sample readings and flags unusual usage. These tasks serve different needs. For example, a forecast helps plan for tomorrow, while a photo can guide an inspection today."*

* **40–60 sec:**  
  > *"Our aim is to compare the models and present outputs people can interpret. The dashboard could show a demand curve, a photo with labelled equipment, or a flagged time period. **Binita Roy** will explain where the data comes from."*

---

## 04 Binita Roy
**Topic:** Datasets and System Architecture  
**Slide Coverage:** Slides 7–8  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"We plan to use smart meter readings from Mathura and Bareilly, grid demand records, field photos from the PD-Defect dataset, and household energy surveys. The readings show how usage changes over time, while the photos show equipment in the field."*

* **20–40 sec:**  
  > *"Each source needs different preparation. We clean meter readings, align timestamps in the grid records, and resize field photos while checking their labels. We then send each input to the model or analysis that suits it."*

* **40–60 sec:**  
  > *"The analytics layer brings the outputs together, and the backend passes them to the dashboard. Users can then view graphs, labelled images, and unusual usage alerts. **Dyutimoy Panda** will explain how the first model estimates household electricity use."*

---

## 05 Dyutimoy Panda
**Topic:** Household Electricity Prediction (ANN)  
**Slide Coverage:** Slide 9  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"ANN stands for artificial neural network. In this module, we use information from smart meter readings to estimate household electricity use. We start with a simple perceptron to show how an artificial neuron makes a basic decision."*

* **20–40 sec:**  
  > *"A model with a hidden layer can learn more complex patterns. We will test a multilayer network for consumption prediction and compare activation functions such as ReLU and sigmoid using the same split of the data."*

* **40–60 sec:**  
  > *"On the sample chart, one line represents actual use and the other shows the prediction. The gap between them illustrates prediction error. We can check where the model misses a rise in usage. **Enakhya Mandal** will explain the training experiments."*

---

## 06 Enakhya Mandal
**Topic:** Deep Neural Network Training (DNN)  
**Slide Coverage:** Slide 10  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"The goal of this module is to compare ways to train a deep neural network. For example, we can test Adam and SGD. Both adjust the model's weights as it learns, but they use different update rules."*

* **20–40 sec:**  
  > *"To make the comparison fair, we keep the model architecture and data split the same and change one training choice at a time. We can repeat this for other choices, such as whether to use dropout."*

* **40–60 sec:**  
  > *"We will record training time and compare validation loss, which measures error on data held out from training. A method may train quickly and still perform poorly on those readings. **Pritam** will now explain the image inspection module."*

---

## 07 Pritam
**Topic:** AI-Based Infrastructure Inspection  
**Slide Coverage:** Slide 11  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"This module uses field photos to help with equipment inspections. The first task is classification, which gives the whole photo a label. For example, a model could help tell whether a pole looks straight or appears to lean."*

* **20–40 sec:**  
  > *"The second task is object detection. It draws a box around equipment so someone can see where it is in the picture. The photo on the right shows labelled boxes around the poles as an example."*

* **40–60 sec:**  
  > *"The third task is segmentation. It marks the pixels that belong to a particular area, such as vegetation near a wire. These outputs can help staff decide which photos to review first. **Rittick Raj Mishra** will explain demand forecasting."*

---

## 08 Rittick Raj Mishra
**Topic:** Electricity Demand Forecasting  
**Slide Coverage:** Slide 12  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"Demand forecasting uses readings in time order to estimate what comes next. In this example, we take seven days of historical electricity use and forecast demand for the next twenty-four hours. The past readings form the model's input."*

* **20–40 sec:**  
  > *"We will compare several sequence models, including LSTM and GRU. Each model gets the same amount of history and is tested on the same later period. This lets us compare how well they handle changes in demand."*

* **40–60 sec:**  
  > *"We will look at overall forecast error and also check whether the model gets the size and timing of the daily peak right. Both matter when planning for busy hours. **Sahil Ahmed** will now explain synthetic data and unusual usage."*

---

## 09 Sahil Ahmed
**Topic:** Synthetic Data and Anomaly Detection (GAN and VAE)  
**Slide Coverage:** Slide 13  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"This slide shows two uses of generative models. In the first, a GAN creates sample electricity readings. One part generates samples, while the other learns to tell generated samples from real readings. That feedback helps improve the generated examples."*

* **20–40 sec:**  
  > *"The second model is a VAE. We train it to reconstruct typical electricity readings. If a new set of readings is difficult to reconstruct, the dashboard can flag that period for a closer look by a person."*

* **40–60 sec:**  
  > *"The generated readings could give us extra examples for model training. The flags could help users find periods of unusual usage without checking every reading by hand. **Supriyo Jana** will finish with the expected results and applications."*

---

## 10 Supriyo Jana
**Topic:** Expected Results, Applications, and Conclusion  
**Slide Coverage:** Slides 14–15  
**Target Time:** 60 Seconds  

* **0–20 sec:**  
  > *"We expect the dashboard to show forecasts, labelled field photos, and unusual usage alerts. A user could compare predicted and actual consumption, see the estimated peak for the day, or look at a photo that needs review."*

* **20–40 sec:**  
  > *"These outputs have different uses. A utility team could use forecasts to prepare for busy hours. An inspection team could use labelled photos to prioritise its work. Households could use consumption charts to understand their own usage patterns."*

* **40–60 sec:**  
  > *"A future step would be to connect live smart meters so the dashboard can show new readings as they arrive. That brings us to the end of our SmartGrid AI presentation. Thank you for listening. We're happy to take questions."*

---

### ⏱️ Rehearsal Tips for the Team
- **Practice with a timer:** Each speaker has three 20-second checkpoints to hit the exact 60-second mark cleanly.
- **Listen for handoff names:** Each speaker explicitly introduces the next teammate in the final 5 seconds (e.g., *"Pritam will now explain the image inspection module"*).

