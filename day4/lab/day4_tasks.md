# Day 4 Lab: Building and Deploying Machine Learning APIs with FastAPI

## Table of Contents

- [Day 4 Lab: Building and Deploying Machine Learning APIs with FastAPI](#day-4-lab-building-and-deploying-machine-learning-apis-with-fastapi)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Theoretical Concepts](#theoretical-concepts)
    - [What is FastAPI?](#what-is-fastapi)
    - [Asynchronous Programming in Python](#asynchronous-programming-in-python)
    - [Why Use FastAPI for ML APIs?](#why-use-fastapi-for-ml-apis)
    - [Containerizing and Deploying FastAPI](#containerizing-and-deploying-fastapi)
  - [Lab Instructions](#lab-instructions)
    - [Part 0: Understanding Async in Python](#part-0-understanding-async-in-python)
      - [Task 0.1: Basic synchronous example](#task-01-basic-synchronous-example)
      - [Task 0.2: Basic async Python example](#task-02-basic-async-python-example)
      - [Task 0.2: Understanding Concurrency with Async Tasks](#task-02-understanding-concurrency-with-async-tasks)
      - [Key Learning Points](#key-learning-points)
    - [Part 1: Setting up FastAPI](#part-1-setting-up-fastapi)
      - [Task 1.1: Installing FastAPI and Uvicorn](#task-11-installing-fastapi-and-uvicorn)
      - [Task 1.2: Setting Up a Basic FastAPI App](#task-12-setting-up-a-basic-fastapi-app)
      - [Task 1.3: Running the FastAPI App on localhost](#task-13-running-the-fastapi-app-on-localhost)
    - [Part 2: Model Training and Setup](#part-2-model-training-and-setup)
      - [Task 2.1: Train and Save Machine Learning Models](#task-21-train-and-save-machine-learning-models)
      - [Task 2.2: Load Models in FastAPI using Lifespan](#task-22-load-models-in-fastapi-using-lifespan)
      - [Task 2.3: Create a GET Endpoint to List Available Models](#task-23-create-a-get-endpoint-to-list-available-models)
      - [Task 2.4: Use a `.env` file for the location of models](#task-24-use-a-env-file-for-the-location-of-models)
    - [`[BONUS]` Part 3: Building a Simple Prediction API](#bonus-part-3-building-a-simple-prediction-api)
      - [Task 3.1: Set Up a POST Prediction Endpoint](#task-31-set-up-a-post-prediction-endpoint)
      - [Task 3.2: Adding Long Asynchronous Predictions](#task-32-adding-long-asynchronous-predictions)
      - [Task 3.3: Enhanced Schema Validation](#task-33-enhanced-schema-validation)
    - [Part 4: Dockerizing and Deploying FastAPI](#part-4-dockerizing-and-deploying-fastapi)
      - [Task 4.1: Dockerizing the Application](#task-41-dockerizing-the-application)
      - [Task 4.2: Deploy to Cloud (GCP)](#task-42-deploy-to-cloud-gcp)
  - [Conclusion](#conclusion)
  - [Useful Links](#useful-links)

## Overview

In this lab, we will build a machine learning API using FastAPI. We'll start by understanding asynchronous programming in Python, followed by building a simple API to serve multiple ML models using FastAPI. We’ll also add schema validation, error handling, and containerize the FastAPI application using Docker, followed by deployment to a cloud platform (GCP or Azure).

## Theoretical Concepts

### What is FastAPI?

**FastAPI** is a modern, fast web framework for building APIs with Python. It is designed to be easy to use and powerful, offering automatic request validation, async support, and more.

- **Main Features**:
  - High performance, comparable to NodeJS and Go.
  - Automatic interactive API documentation (Swagger UI and ReDoc).
  - Built-in support for asynchronous code.
  - Automatic validation of request data using **Pydantic**.

### Asynchronous Programming in Python

**Asynchronous programming** allows Python programs to handle multiple requests concurrently. Using `async` and `await`, you can write non-blocking code that performs tasks in parallel without waiting for one task to finish before starting another.

For example, when an ML model takes time to predict, async allows the API to handle other requests while waiting for the prediction to finish.

### Why Use FastAPI for ML APIs?

FastAPI is ideal for machine learning APIs because of:

- **Speed**: FastAPI is designed for high-performance applications.
- **Async Support**: Handle multiple requests simultaneously, which is crucial when serving large models or performing time-consuming inference tasks.
- **Built-in Validation**: FastAPI uses **Pydantic** for request validation, ensuring correct data input.
- **Ease of Deployment**: FastAPI can be easily deployed using Docker and cloud platforms like **GCP** and **Azure**.

### Containerizing and Deploying FastAPI

Using **Docker**, you can create lightweight containers that package your API and its dependencies, making it portable across different environments. Once containerized, you can deploy your FastAPI app to cloud services like **Google Cloud Run** or **Azure App Services**.

## Lab Instructions

### Part 0: Understanding Async in Python

Before diving into FastAPI, let's explore the concepts of synchronous and asynchronous programming in Python. We'll show how asynchronous tasks can help reduce wait times when handling multiple tasks concurrently.

**Goal**: By the end of this section, you will understand how Python’s async and await keywords allow you to run tasks concurrently, improving efficiency in scenarios where tasks involve waiting (e.g., I/O-bound tasks).

#### Task 0.1: Basic synchronous example

Run the following code and note how Task 2 waits for Task 1 to finish before starting:

  ```python
  import time

  def slow_task(name, delay):
      print(f"Starting task {name}: {time.strftime('%X')}")
      time.sleep(delay)
      print(f"Finished task {name}: {time.strftime('%X')}")

  def main():
      slow_task("Task 1", 3)
      slow_task("Task 2", 7)
      print("All tasks finished")

  main()
  ```

#### Task 0.2: Basic async Python example

Now, let's convert the previous example into an asynchronous one. Even though this is asynchronous, we're still awaiting each task one after another.

**Explanation**:

- The `async def` keyword marks the function as asynchronous.
- The `await` keyword allows the function to "pause" without blocking other tasks.

  ```python
  import asyncio
  import time

  async def slow_task(name, delay):
      print(f"Starting task {name}: {time.strftime('%X')}")
      await asyncio.sleep(delay)
      print(f"Finished task {name}: {time.strftime('%X')}")

  async def main():
      await slow_task("Task 1", 3)
      await slow_task("Task 2", 7)
      print("All tasks finished")

  asyncio.run(main())
  ```

#### Task 0.2: Understanding Concurrency with Async Tasks

Let's now modify the code to run both tasks concurrently.

**Explanation**:

- `asyncio.create_task()` schedules the tasks to run concurrently.
- The program doesn’t wait for `task1` to complete before starting `task2`. Instead, both tasks are "created" at nearly the same time and will run concurrently.

  ```python
  import asyncio
  import time

  async def slow_task(name, delay):
      print(f"Starting task {name}: {time.strftime('%X')}")
      await asyncio.sleep(delay)
      print(f"Finished task {name}: {time.strftime('%X')}")

  async def main():
      print(f"Starting: {time.strftime('%X')}")

      task1 = asyncio.create_task(slow_task("Task 1", 3))
      print(f"Created task 1: {time.strftime('%X')}")

      task2 = asyncio.create_task(slow_task("Task 2", 7))
      print(f"Created task 2: {time.strftime('%X')}")

      await task1
      print(f"Awaited task 1: {time.strftime('%X')}")

      await task2
      print(f"Awaited task 2: {time.strftime('%X')}")

  asyncio.run(main())
  ```

#### Key Learning Points

- Synchronous programming: Tasks are executed one after another, causing delays when tasks involve waiting.
- Asynchronous programming: Tasks run concurrently, reducing overall execution time, especially when waiting for I/O operations.

### Part 1: Setting up FastAPI

This section will guide you through setting up a FastAPI app with a simple health check endpoint.

#### Task 1.1: Installing FastAPI and Uvicorn

- Install FastAPI and Uvicorn using `pip`.

#### Task 1.2: Setting Up a Basic FastAPI App

- Create a new Python file and set up a basic FastAPI app (the root endpoint `/` should return "Hello world").
- Add a simple GET endpoint `/health` that returns a JSON response indicating the health status "up and running".

#### Task 1.3: Running the FastAPI App on localhost

- Run the FastAPI app on `localhost` using `uvicorn` or `fastapi`.
- Test the `/health` endpoint to verify it is working.
- Explore the interactive API documentation at `/docs`.

---

### Part 2: Model Training and Setup

In this part, you will train and save two machine learning models and load them into your FastAPI app.

#### Task 2.1: Train and Save Machine Learning Models

- Update the `model_training.py` script so that the models get written to disk.

#### Task 2.2: Load Models in FastAPI using Lifespan

- Look into the `lifespan` feature of FastAPI.
- Implement FastAPI’s `lifespan` feature to load the machine learning models when the app starts. For example, you can use an empty global object that is initiated in the `lifespan` function to mimic "registering/loading models".

#### Task 2.3: Create a GET Endpoint to List Available Models

- Add a new GET endpoint `/models` that returns the names of the loaded models.

#### Task 2.4: Use a `.env` file for the location of models

We will now create a `.env` file where we will store the location (path) to our models.

1. Create a `.env` file and add two lines in it:

   ```
   LOGISTIC_MODEL={PATH_TO_YOUR_LOGISITC_MODEL}
   RF_MODEL={PATH_TO_YOUR_RF_MODEL}
   ```

2. Load the `.env` varibles and access them in code.

   - Install the required package.

   - Load the `.env` variables in code.

   - Access the loaded environemnt variables where needed.

---

### `[BONUS]` Part 3: Building a Simple Prediction API

In this section, you will create a prediction API to serve machine learning models using FastAPI.

#### Task 3.1: Set Up a POST Prediction Endpoint

- Define a POST endpoint `/predict/{model_name}` that accepts input data in JSON format.
- Use Pydantic to validate the input data.
- Make predictions using the loaded ML models based on the input data.

#### Task 3.2: Adding Long Asynchronous Predictions

- Simulate a long-running operation using `await asyncio.sleep()` before returning the prediction.
- Test the asynchronous behavior by sending multiple requests and ensuring they run concurrently.

#### Task 3.3: Enhanced Schema Validation

- Enhance your Pydantic model to validate input data more rigorously.
  - Add constraints like minimum and maximum values for each input feature.
  - Include descriptions and example values for each feature.

---

### Part 4: Dockerizing and Deploying FastAPI

Now that your FastAPI app is ready, we will Dockerize and deploy it to a cloud platform.
Make sure you have installed and configured docker on your machine: [tutorial](https://docs.docker.com/get-started/get-docker/).

#### Task 4.1: Dockerizing the Application

- Write a `Dockerfile` to containerize your FastAPI app.
- Build the Docker image locally and test the containerized app (run the image).

#### Task 4.2: Deploy to Cloud (GCP)

- Deploy your Dockerized FastAPI app to **Google Cloud Platform**.
  - Push the Docker image to the appropriate container registry.
  - Deploy the container to **Cloud Run** (GCP)

---

## Conclusion

In this lab, you explored the following topics:

- Synchronous vs. Asynchronous programming in Python
- Building an API with FastAPI
- Serving machine learning models through an API
- Validating input data with Pydantic
- Dockerizing and deploying a FastAPI app to the cloud

This lab prepares you for building scalable machine learning APIs in production environments.

---

## Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Azure Container Registry Documentation](https://docs.microsoft.com/en-us/azure/container-registry/)
