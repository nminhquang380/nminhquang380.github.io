---
layout: post
title: "Tools for Data Science"
date: 2023-08-31 20:35:00 +1000
categories: machinelearning 
---
## Module 1: Overview of Data Science
- The Data Science Task Categories include:
    - Data Management -  storage, management and retrieval of data
    - Data Integration and Transformation - streamline data pipelines and automate data processing tasks
    - Data Visualization - provide graphical representation of data and assist with communicating insights
    - Modelling - enable Building, Deployment, Monitoring and Assessment of Data and Machine Learning models

- Data Science Tasks support the following:
    - Code Asset Management - store & manage code, track changes and allow collaborative development
    - Data Asset Management - organize and manage data, provide access control, and backup assets
    - Development Environments - develop, test and deploy code
    - Execution Environments - provide computational resources and run the code

The data science ecosystem consists of many open source and commercial options, and include both traditional desktop applications and server-based tools, as well as cloud-based services that can be accessed using web-browsers and mobile interfaces.

**Data Management Tools**: include Relational Databases, NoSQL Databases, and Big Data platforms:

- MySQL, and PostgreSQL are examples of Open Source Relational Database Management Systems (RDBMS), and IBM Db2 and SQL Server are examples of commercial RDBMSes and are also available as Cloud services.
- MongoDB and Apache Cassandra are examples of NoSQL databases.
- Apache Hadoop and Apache Spark are used for Big Data analytics. 

**Data Integration and Transformation Tools**: include Apache Airflow and Apache Kafka. 

**Data Visualization Tools**:  include commercial offerings  such as Cognos Analytics, Tableau and PowerBI  and can be used for building dynamic and interactive dashboards.  

**Code Asset Management Tools**: Git is an essential code asset management tool. GitHub is a popular web-based platform for storing and managing source code. Its features make it an ideal tool for collaborative software development, including version control, issue tracking, and project management. 

**Development Environments**: Popular development environments for Data Science include Jupyter Notebooks and RStudio. 
- Jupyter Notebooks provides an interactive environment for creating and sharing code, descriptive text, data visualizations, and other computational artifacts in a web-browser based interface.  
- RStudio is an integrated development environment (IDE) designed specifically for working with the R programming language, which is a popular tool for statistical computing and data analysis.  

## Module 2: Languages of Data Science
- The popular languages are Python, R, SQL, Scala, Java, C++, and Julia.
- For data science, you can use Python's scientific computing libraries like Pandas, NumPy, SciPy, and Matplotlib. 
- Python can also be used for Natural Language Processing (NLP) using the Natural Language Toolkit (NLTK). 
- Python is open source, and R is free software. 
- R language’s array-oriented syntax makes it easier to translate from math to code for learners with no or minimal programming background.
- Python is open-sourse, while R is free software. They are different.
- SQL is different from other software development languages because it is a non-procedural language.
- SQL was designed for managing data in relational databases. 
- If you learn SQL and use it with one database, you can apply your SQL knowledge with many other databases easily.
- Data science tools built with Java include Weka, Java-ML, Apache MLlib, and Deeplearning4.
- For data science, popular program built with Scala is Apache Spark which includes Shark, MLlib, GraphX, and Spark Streaming.
- Programs built for Data Science with JavaScript include TensorFlow.js and R-js.
- One great application of Julia for Data Science is JuliaDB.

## Module 3: Libraries, APIs, Datasets and Models
### Libraries are a collection of functions and methods
#### Python Libraries:
1. Scientifics Computing Libraries:
    - Pandas (Data Structures & Tools)
    - Numpy (Arrays & Matrices), Pandas on top of NumPy
2. Visualization:
    - Matplotlib (plots & Graphs)
    - Seaborn (heat maps, time series, violin plots)
3. Machine Learning and Deep Learning
    - Scikit-learn (Machine Learning)
    - Keras (Deep Learning Neural Networks)
    - TensorFlow (low-level framework for large scale)
    - PyTorch (simpler)

#### Apache Spark
>> General-purpose cluster-computing framework: Pandas, Numpy, Sklearn
- Scala libraries:
    - Statistical Visualization: VEGAS
    - Deep learning: Big DL
- R libraries:
    - Visualization: ggplot2

### API
- REST APIs (Representational State Transfer APIs)
    - Allow to communicate through the internet. Also have a set of rules.
    - Enable you to use resources like storage, data, and artificially intelligent algorithms.
    - Use HTTP method.

### Dataset
- Collection of data
- Data Structures
    - Tabular Data (contain number of rows, ex: csv,..)
    - Hierachical data, network data.
    - Raw files.
- Private data
    - Confidential
    - Private or personal infomation
    - Commercially sensitive
- Open data
    - Publicly available
    - Companies
    - Scientific institutions
    - Government
- Community Data License Agreement

### Machine Learning Models
- Types of ML are Supervised, Unsupervised, and Reinforcement. 
- Supervised learning comprises two types of models, regression and classification.
- Deep learning refers to a general set of models and techniques that loosely emulate the way the human brain solves a wide range of problems.
- The Model Asset eXchange is a free, open-source repository for ready-to-use and customizable deep-learning microservices.
- MAX model-serving microservices are built and distributed on GitHub as open-source Docker images.
- You can use Red Hat OpenShift, a Kubernetes platform, to automate deployment, scaling, and management of microservices.
- Ml-exchange.org has multiple predefined models.

## Module 4: Jupyter Notebooks and JupyterLab
- Jupyter Notebooks are used in Data Science for recording experiments and projects.
- Jupyter Lab is compatible with many files and Data Science languages.
- There are different ways to install and use Jupyter Notebooks.
- How to run, delete, and insert a code cell in Jupyter Notebooks.
- How to run multiple notebooks at the same time.
- How to present a notebook using a combination of Markdown and code cells.
- How to shut down your notebook sessions after you have completed your work on them.
- Jupyter implements a two-process model with a kernel and a client.
- The notebook server is responsible for saving and loading the notebooks.
- The kernel executes the cells of code contained in the Notebook. 
- The Jupyter architecture uses the NB convert tool to convert files to other formats.
- Jupyter implements a two-process model with a kernel and a client.
- The Notebook server is responsible for saving and loading the notebooks.
- The Jupyter architecture uses the NB convert tool to convert files to other formats. 
- The Anaconda Navigator GUI can launch multiple applications on a local device.
- Jupyter environments in the Anaconda Navigator include JupyterLab and VS Code.
- You can download Jupyter environments separately from the Anaconda Navigator, but they may not be configured properly.
- The Anaconda Navigator GUI can launch multiple applications.
- Additional open-source Jupyter environments include JupyterLab, JupyterLite, VS Code, and Google Colaboratory. 
- JupyterLite is a browser-based tool.
