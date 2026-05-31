# Machine Learning Engineering on AWS — Second Edition<br />Build, deploy, and operationalize LLMs, AI agents, and generative AI systems on AWS

Welcome to the official GitHub repository for *Machine Learning Engineering on AWS — Second Edition* by [Joshua Arvin Lat](https://www.amazon.com/author/arvs), published by Packt. Here you'll find the commands and code blocks referenced throughout the [book](https://www.packtpub.com/en-us/product/machine-learning-engineering-on-aws-9781835881088).

![Machine Learning Engineering on AWS — Second Edition](books.png)

This book is for AI engineers, data scientists, machine learning engineers, and technology leaders who want to deepen their understanding of machine learning engineering, generative AI, large language models, retrieval-augmented generation, AI agents, and MLOps on AWS. A foundational understanding of artificial intelligence, machine learning, generative AI, and cloud engineering concepts is recommended.

A lot has changed since I wrote the first edition of this book. Back then, generative AI was still emerging, and many organizations were only beginning to explore how large language models (LLMs) could change the way we build machine learning systems and workflows. Today, generative AI has become a core part of real-world applications, which means building modern AI systems now requires much more than just training models. 

![Machine Learning Engineering on AWS — Second Edition](book-cover.png)

It also involves production engineering, LLMOps automation, security, evaluation, and scalable cloud-based architectures. In this second edition, I want to help you understand how these modern AI systems are built on AWS through practical, hands-on examples covering generative AI, AI agents, data engineering, model deployment, evaluation, and automation, so you can confidently design and operate production-ready AI solutions.


## Where to Get Your Copy

You can secure your copy of *Machine Learning Engineering on AWS — Second Edition* from major online retailers such as [Amazon](https://amazon.com/author/arvs) or directly from the publisher, [Packt](https://www.packtpub.com/en-us/product/machine-learning-engineering-on-aws-9781835881088). Choose the format that works best for you. 🙏


## Chapter 1: A Gentle Introduction to Generative AI and AI Agents on AWS

In this chapter, you'll explore the fundamentals of generative AI on AWS and learn how to use various services and solutions to build AI agents. You will work with foundation models provided through Amazon Bedrock, cover key concepts and terminology, set up a SageMaker Studio space, and build your first AI agent using Strands Agents with tool integrations to enhance reasoning and problem-solving capabilities.

We will cover the following topics in this chapter:

- Generative AI for the modern machine learning engineer
- Exploring foundation models in Amazon Bedrock
- Setting up and configuring your SageMaker Studio environment
- Configuring IAM permissions for your SageMaker Studio Space
- Introduction to AI agents with Amazon Bedrock and Strands Agents


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter01">
        CHAPTER 01 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 2: Building AI Agents with SageMaker AI and Bedrock AgentCore

In this chapter, you'll learn how to build AI agents that interact with a SageMaker AI real-time inference endpoint. You will use Amazon Bedrock Knowledge Bases and Amazon S3 Vectors to build retrieval-augmented generation powered agents, while also exploring how Strands Agents and Bedrock AgentCore can integrate model inference, external tools, and knowledge retrieval into production-ready agent-based systems.

To help you gain hands-on experience running AI agents with SageMaker AI, Strands Agents, and Bedrock AgentCore, we will cover the following topics in this chapter:

- Deploying a pretrained LLM with SageMaker AI
- Building AI agents with Amazon SageMaker AI and Strands Agents
- Building AI agents with Amazon Bedrock AgentCore
- Deploying production-ready agents with Bedrock AgentCore Runtime
- Setting up an Amazon Bedrock Knowledge Base
- Building a RAG-powered AI agent with Strands Agents
- Building a RAG-powered AI agent that interacts with a SageMaker AI inference endpoint


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter02">
        CHAPTER 02 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 3: Machine Learning Engineering with Amazon SageMaker AI

In this chapter, you'll learn the fundamentals of machine learning engineering on AWS and use Amazon SageMaker AI to implement end-to-end machine learning workflows. You will train and deploy an XGBoost model, fine-tune a BERT model, and explore how SageMaker AI simplifies training, inference, and model lifecycle management through managed capabilities.

To help you explore how you can use Amazon SageMaker AI to implement end-to-end ML engineering workflows involving both traditional ML models and foundation models, this chapter covers the following sections:

- Setting up and preparing your JupyterLab notebook
- Preparing a synthetic dataset for binary classification
- Training an XGBoost binary classifier
- Deploying an XGBoost model to a real-time inference endpoint
- Setting up BERT fine-tuning with SageMaker JumpStart
- Using a smaller dataset for fine-tuning
- Running the BERT model fine-tuning job
- Deploying the fine-tuned model to a real-time inference endpoint


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter03">
        CHAPTER 03 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 4: Modernizing Analytics with a Managed Transactional Data Lake

In this chapter, you'll build and work with a transactional data lake using Amazon S3 tables. You will create an Amazon S3 table bucket, launch an Amazon EMR cluster with Apache Iceberg, run queries using Apache Spark, and explore time travel queries to analyze how datasets evolve over time.

We will cover the following topics in this chapter:

- Preparing and processing the synthetic data
- Creating an Amazon S3 table bucket
- Launching an Amazon EMR cluster with Apache Iceberg installed
- Performing Apache Iceberg queries on S3 tables with Apache Spark
- Performing time travel queries on S3 tables


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter04">
        CHAPTER 04 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 5: Practical Data Management on AWS

In this chapter, you'll explore AWS services and capabilities that support data management for analytics and machine learning workloads. You will work with AWS Lake Formation permissions, query data using Amazon Athena, ingest data into Amazon SageMaker Feature Store, and retrieve data from both the online and offline feature stores.

To help you build practical data management skills for modern cloud-based ML workflows, we will cover the following topics in this chapter:

- Working with AWS Lake Formation permissions
- Running SQL queries in Amazon Athena
- Ingesting data into a SageMaker feature store
- Adding searchable metadata to the features
- Retrieving data from the online and offline feature stores


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter05">
        CHAPTER 05 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 6: Pragmatic Data Processing on AWS

In this chapter, you'll learn how to use SageMaker Processing jobs for resource-intensive data processing workloads. You will run a back-translation workflow using SageMaker Processing, prepare datasets and scripts, and explore best practices for designing, managing, scaling, and securing data processing workflows.

We will cover the following topics in this chapter:

- Getting started with SageMaker Processing jobs
- Running your first SageMaker Processing job
- Preparing the input data and script for the back translation job
- Automating back translation workflows with SageMaker Processing jobs


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter06">
        CHAPTER 06 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 7: SageMaker AI Model Training and Tuning Capabilities

In this chapter, you'll fine-tune a large language model using Amazon SageMaker AI as part of an end-to-end machine learning workflow. You will track experiments using MLflow, execute supervised fine-tuning jobs, perform hyperparameter tuning to identify the best-performing model, and deploy the final model to a real-time inference endpoint.

We will cover the following topics in this chapter:

- Setting up a serverless MLflow App
- Fine-tuning an LLM on Amazon SageMaker AI
- Deploying the Fine-Tuned Model
- Performing Hyperparameter Tuning with Amazon SageMaker AI
- Deploying the Best-Performing Model from Hyperparameter Tuning


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter07">
        CHAPTER 07 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 8: SageMaker AI Model Deployment Options and Strategies

In this chapter, you'll explore different deployment options and strategies in Amazon SageMaker AI. You will deploy models using real-time, serverless, asynchronous, and batch inference options, while also practicing advanced deployment techniques such as shadow testing, canary traffic shifting, and inference data capture for monitoring and evaluation.

This chapter covers the following topics:

- Preparing your JupyterLab Notebook for model deployment
- Deploying your model to a real-time inference endpoint
- Deploying your model to a serverless inference endpoint
- Running batch inference with batch transform
- Deploying your model to an asynchronous inference endpoint
- Setting up a shadow test with a SageMaker inference endpoint
- Using canary traffic shifting when performing blue/green deployments


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter08">
        CHAPTER 08 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>

## Chapter 9: Automating LLMOps Workflows with SageMaker Pipelines

In this chapter, you'll design and operationalize LLMOps pipelines using SageMaker Pipelines. You will build single-step and multi-step workflows, integrate AWS Lambda-based orchestration steps, and learn best practices for building scalable, maintainable, secure, and cost-efficient production-grade machine learning pipelines.

This chapter covers the following topics:

- Setting up the project environment and dependencies
- Building and Running the Single-Step Fine-Tuning Pipeline
- Building and Running the Single-Step Evaluation Pipeline
- Configuring and Running a Two-Step Fine-Tuning and Evaluation Pipeline
- Preparing the Lambda functions for deploying a model to an endpoint
- Completing the LLMOps pipeline
- Best Practices and Key Considerations for Building Automated ML Workflows


**Chapter Resources**:

<table>
  <tr>
    <td>
      &nbsp;➤&nbsp;&nbsp;&nbsp;
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter09">
        CHAPTER 09 — README.md
      </a>
      &nbsp;&nbsp;
    </td>
  </tr>
</table>


## What’s new in the second edition that wasn’t covered in the previous one?

A lot has changed since I wrote the first edition of the book. At the time, generative AI was still emerging, and most organizations were only beginning to explore what large language models could do and how they might influence the way you build applications and machine learning workflows. Today, generative AI is being integrated into real production systems, with LLMs, retrieval-augmented generation, and AI agents becoming part of everyday applications across many industries.

![Amazon A+ 01](a-plus/a-plus-01.png)

This second edition has been rewritten from the ground up to reflect how dramatically the AI landscape has evolved in recent years. This edition explores practical generative AI systems on AWS in much greater depth. You’ll work with services and solutions such as Amazon Bedrock, SageMaker AI, Bedrock AgentCore, and Strands Agents while building RAG-powered applications, AI agents, and automated workflows. You’ll also explore newer topics that have become increasingly important, including LLM-as-a-judge evaluation techniques, end-to-end automation with SageMaker Pipelines, and advanced model fine-tuning and deployment using SageMaker AI.

## Why is now the right time for engineers to master generative AI and LLMOps on AWS?

The demand for AI-powered applications is growing faster than most technology shifts in the past decade. Now is the right time for engineers to explore building generative AI systems on AWS, as AI is becoming central to how modern applications are designed and built. Many engineers focus on AI engineering at the application level, while more advanced practitioners go deeper into machine learning engineering and LLMOps to understand how models are fine-tuned, deployed, evaluated, and managed in production.

![Amazon A+ 02](a-plus/a-plus-02.png)

Mastering generative AI and LLMOps is not something that happens overnight. In fact, it may take decades of continuous learning and hands-on experience to develop deep expertise across the many disciplines involved. The good news is that you do not need to master everything before you start building. Every project, experiment, and deployment helps you develop the skills needed to work effectively with these technologies in real-world environments.

## What makes this book different from other MLOps or generative AI books available today?

This book takes you through a practical, end-to-end process of building and managing generative AI systems and workflows on AWS. This book focuses on building complete, production-ready systems that combine AI agents, RAG-powered systems, and MLOps workflows in a unified and practical way. Each chapter is designed to equip you with the knowledge and skills you need, reinforcing what you learned previously while introducing new concepts that help you build working systems from end to end.

![Amazon A+ 03](a-plus/a-plus-03.png)

I also assume you may be learning these services for the first time, so I introduce concepts gradually before moving into more advanced patterns. As you go through each chapter, you will learn how to troubleshoot common problems, apply best practices, and address real-world production requirements. I wanted this book to feel like the kind of book I wish I had when I was just getting started with AI, machine learning, and machine learning engineering. Back then, I often had to piece things together from scattered resources, and it was not always clear where to begin. This book is meant to give you that starting point and make the learning path more straightforward for generative AI, machine learning engineering, and MLOps.

## What will readers be able to build or deploy in real-world environments after reading this book?

After reading this book, you will be able to build and deploy real-world generative AI systems on AWS from end to end. You will build AI agents using Amazon Bedrock AgentCore and Strands Agents that interact with large language models deployed in a managed inference endpoint, enabling them to reason and respond in real applications. You will also build RAG-powered systems using Amazon Bedrock Knowledge Bases and S3 Vectors, allowing your applications to answer questions grounded in your own data.

![Amazon A+ 04](a-plus/a-plus-04.png)

Along the way, you will also learn how to work like a machine learning engineer in production. You will fine-tune and deploy models using Amazon SageMaker AI, automate LLMOps workflows with SageMaker Pipelines, and evaluate model behavior using LLM-as-a-judge techniques. By the end, you will be able to combine these building blocks to design and engineer more complex AI-powered systems that are ready for real-world production use.

## How do you guide readers in building a complete RAG-powered application from data to deployment on AWS?

In the second chapter of this book, I will walk you through step by step how to build a complete RAG-powered application on AWS. I will guide you the same way you might learn to build a miniature bridge on a beach. You will begin by understanding how each part works and gradually integrate these building blocks into something more complete and functional.

![Amazon A+ 05](a-plus/a-plus-05.png)

You will start with a sample dataset and upload it to Amazon S3. From there, you will learn how Amazon Bedrock Knowledge Bases will ingest the data, generate vector embeddings, and store them in a managed vector store such as Amazon S3 Vectors or other supported vector databases to make the content searchable. You will then connect this knowledge layer to a RAG-powered AI agent which reasons over the retrieved context and generates responses.

## Get to Know the Author

**Joshua Arvin Lat** serves as the Vice President of Cybersecurity and AI for **Axos**. He previously held Chief Technology Officer and Director roles across SaaS platforms, AI automation companies, e-commerce startups, and digital agencies. Because of his proven track record in leading digital transformation within organizations, he has been recognized as one of the winners of the prestigious Orange Boomerang: Digital Leader of the Year 2023 award. 

![Machine Learning Engineering on AWS 2nd ed](arvs-machine-learning-engineering-on-aws.png)

Years ago, he led a team that won first place in a global cybersecurity competition for their published research. He is also an AWS AI Hero and has spoken at several international conferences on practical applications of generative AI, software engineering, cybersecurity, and management. To give back to the technology community, he has authored five books covering artificial intelligence, cloud computing, and cybersecurity.

![The author's books](5-books.png)

You can find the author's other books on AI and Cybersecurity by visiting the [Amazon Author Page](https://amazon.com/author/arvs)
