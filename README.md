# What is Raasta?

Raasta is an undergraduate final-year project developed at Habib University. This project not only demonstrates the technical skills and expertise we have accumulated throughout the duration of our studies at Habib University but also addresses a critical issue that is regularly being faced by Karachi's citizens- rapidly deteriorating road quality. Our website that showcases this work is available at [Raasta](https://raasta-web.web.app). 

_Note: This is a personal repository meant to showcase our results for potential scaling, and is not meant for reproduction._

# Why Raasta?

Karachi's roads have been continuously neglected due to insufficient funding and mismanagement by local government officials, making it an increasingly challenging and dangerous experience for commuters and transporters alike. 

To address this issue, we have developed a road surface classification system using mobile sensor data. Leveraging the tri-axial accelerometer, gyroscope, and GPS receiver, road surface data is collected and used to train a Machine Learning model. The system utilizes a custom CNN-BiLSTM architecture for multi-class classification to detect potholes and other road surface anomalies, providing their location information through a Flutter web application. 

This system hopes to significantly improve driver safety and route planning in Karachi and other cities facing inadequate road maintenance. This system can benefit government officials by providing access to a comprehensive database of detected road surface anomaly location points, enabling them to identify and prioritize areas that require immediate repair and maintenance. By using this system, officials can effectively allocate resources towards road maintenance and management, reducing the number of accidents and improving road conditions for everyone. 


# System Breakdown
### 1. Data Collection Application - Abeer Khan
The folder titled [Sensor Application](https://github.com/AkeelMedina22/Raasta/tree/main/Sensor%20Application) contains an Android-based data collection application, developed using Kotlin. This application enables individuals to collect data on their daily commutes, which can provide valuable insights into the current state of road conditions in Karachi. The collected sensor data is transmitted and stored in a non-relational database hosted on Firebase, which serves as a central repository for all the collected sensor data, facilitating easy access and analysis of the data.

The sensor data collected includes various road conditions such as potholes, speed breakers, traffic, and poor road quality. The application features a tri-axial accelerometer and gyroscope integration, along with GPS receiver usage to determine the smartphone's location during the commute at specified intervals. These features allow the application to accurately detect the conditions a vehicle is traveling through and its position. API's like SensorManager and FusedLocationProviderClient are utilized in the application.

### 2. API - Abeer Khan, Akeel Ather Medina
To access relevant information from the cloud-hosted database, a Flask API is developed and deployed, present at [API](https://github.com/AkeelMedina22/Raasta/tree/main/Website/API).

### 3. Website - Zoha Ovais Karim, Samrah Sahto, Abeer Khan
In order to facilitate the visualization of the results obtained from the data collected on road anomalies in Karachi, a Flutter-based website has been designed at [API](https://github.com/AkeelMedina22/Raasta/tree/main/Website/Flutter%20Website). This website enables users to view and interact with a map of the city, which displays the results in the form of color-coded markers. The Flask API is responsible for providing the necessary data for the visualization. Through this website, users can generate routes and inspect them by observing the placement of the markers on the map. The markers serve as an indication of the road conditions along the given route, allowing users to make informed decisions regarding their travel plans. 

### 4. ML Models - Akeel Ather Medina
To process mobile sensor data into useful road-surface information, a variety of Machine Learning models were applied for multi-class classification. As accelerometer data is a function of time, it can be processed as a signal. A variety of preprocessing techniques were applied to filter and reorient the data, followed by synthetic augmentation and resampling. Many ML techniques such as Random Forest, SVM, CNN's were applied, but the best performing model was a residual CNN-BiLSTM where the residual layer learns much larger features than the default layer. The layers were concatenated using spatial pooling. The final results were approximately 85% accuracy on the Deep Learning model, compared to a 65% for Random Forest and SVM. A research paper based on this model was written and is in the process of being published, and the model is available at [ML Model](https://github.com/AkeelMedina22/Raasta/tree/main/ML%20Model).

# Team activity 

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Top contributors
[![Top contributors](https://images.repography.com/33913467/AkeelMedina22/Raasta/top-contributors/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_table.svg)](https://github.com/AkeelMedina22/Raasta/graphs/contributors)


## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Recent activity [![Time period](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_badge.svg)](https://repography.com)
[![Timeline graph](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_timeline.svg)](https://github.com/AkeelMedina22/Raasta/commits)
[![Top contributors](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_users.svg)](https://github.com/AkeelMedina22/Raasta/graphs/contributors)
