# What is Raasta?

Raasta is an undergraduate final-year project developed at Habib University. This project not only demonstrates the technical skills and expertise we have accumulated throughout the duration of our studies at Habib University but also addresses a critical issue that is regularly being faced by Karachi's citizens. Our website is available at [Raasta](https://raasta-web.web.app).

# Why Raasta?

Karachi's roads have been continuously neglected due to insufficient funding and mismanagement by local government officials, making it an increasingly challenging and dangerous experience for commuters and transporters alike. In 2017, road damage was responsible for approximately [10,500](https://www.thenews.com.pk/print/930042-bumpy-rides) accidents in Karachi, and deteriorating roads are to blame for many reported accidents in Pakistan, with almost [55,000](https://www.thenews.com.pk/print/933362-road-safety) people losing their lives in road accidents over the past ten years. Worsening road conditions not only greatly reduce the operational lifetime of automobiles but also pose a severe threat to drivers' safety. with heavy vehicles and monsoon rainfalls greatly exacerbating the problem over the years. Additionally, the physical health of individuals is also affected by poor road infrastructure, as approximately [56%](https://www.thenews.com.pk/print/977712-dilapidated-roads-of-karachi-causing-joint-issues-for-motorcyclists) of young motorcyclists in Karachi suffer from chronic lower back pain as a result of injuries sustained due to poor road conditions, further underscoring the urgent need for effective road maintenance. 

To address this issue, we have developed a road surface classification system using mobile sensing technology. Leveraging smartphone sensors like a tri-axial accelerometer, gyroscope, and GPS receiver, road surface data is collected and used to train a deep learning model. The system utilizes deep learning for multi-class classification to detect potholes and other road surface anomalies and provide their location information through a Flutter web application. This system hopes to significantly improve driver safety and route planning in Karachi and other cities facing similar road maintenance challenges, aiding commuters and drivers in making better-informed choices to ensure a safer and more comfortable traveling experience. This system can benefit government officials by providing access to a comprehensive database of detected road surface anomaly location points, enabling them to identify and prioritize areas that require immediate repair and maintenance. By using this system, officials can effectively allocate resources towards road maintenance and management, reducing the number of accidents and improving road conditions for everyone. 


# Components of Raasta
### 1. Data Collection Application
The current state of research in the field of smartphone sensor technology highlights a significant gap in the availability of a comprehensive data set for the detection of road anomaly locations in Karachi. This lack of data has made it necessary to create a labeled and detailed data set from scratch. To address this issue, an Android-based data collection application is developed using Kotlin. This application enables individuals to collect data on their daily commutes, which can provide valuable insights into the current state of road conditions in Karachi. The collected sensor data is transmitted and stored in a non-relational database hosted on Firebase, which serves as a central repository for all the collected sensor data, facilitating easy access and analysis of the data.

The sensor data collected includes various road conditions such as potholes, speed breakers, traffic, and poor road quality. The application features a tri-axial accelerometer and gyroscope integration, along with GPS receiver usage to determine the smartphone's location during the commute at specified intervals. These features allow the application to accurately detect the conditions a vehicle is traveling through and its position. API's like SensorManager and FusedLocationProviderClient are utilized in the application.

### 2. API
To access relevant information from the cloud-hosted database, a Flask API is developed and deployed.

### 3. Website
In order to facilitate the visualization of the results obtained from the data collected on road anomalies in Karachi, a Flutter-based website has been designed. This website enables users to view and interact with a map of the city, which displays the results in the form of color-coded markers. The Flask API is responsible for providing the necessary data for the visualization. Through this website, users can generate routes and inspect them by observing the placement of the markers on the map. The markers serve as an indication of the road conditions along the given route, allowing users to make informed decisions regarding their travel plans. 

### 4. Deep Learning Model

# Team activity 

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Top contributors
[![Top contributors](https://images.repography.com/33913467/AkeelMedina22/Raasta/top-contributors/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_table.svg)](https://github.com/AkeelMedina22/Raasta/graphs/contributors)


## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Recent activity [![Time period](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_badge.svg)](https://repography.com)
[![Timeline graph](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_timeline.svg)](https://github.com/AkeelMedina22/Raasta/commits)
[![Top contributors](https://images.repography.com/33913467/AkeelMedina22/Raasta/recent-activity/yK18Sv6uzbamK-aXULYcvMWr69C9vCqValaVMgNWBtA/JOTiRrHOifmd6AWoF6yKsXcB81oLiJ-zF2vFxH8pdUQ_users.svg)](https://github.com/AkeelMedina22/Raasta/graphs/contributors)


# Team Contributions
1. Conceptualization, A.M. and A.K.
2. methodology, A.M. and A.K.;
3. software, A.M., A.K., S.A.S., and Z.O.K.;
4. validation, A.M. and A.K.; 
5. formal analysis, A.M.; 
6. investigation, A.M. and A.K.; 
7. data curation, A.M. and A.K.; 
8. writing---original draft preparation, A.M., S.A.S., and Z.O.K.; writing---review and editing, A.M. and A.K., S.A.S., and Z.O.K.; 
9. visualization, A.M.,  A.K., S.A.S., and Z.O.K.; 
10. supervision, A.M. and A.K.; 
11. project administration, A.M. and A.K., S.A.S., Z.O.K. 

