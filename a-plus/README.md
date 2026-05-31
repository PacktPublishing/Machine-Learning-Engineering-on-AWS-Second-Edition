## What’s new in the second edition that wasn’t covered in the previous one?

A lot has changed since I wrote the first edition of the book. At the time, generative AI was still emerging, and most organizations were only beginning to explore what large language models could do and how they might influence the way you build applications and machine learning workflows. Today, generative AI is being integrated into real production systems, with LLMs, retrieval-augmented generation, and AI agents becoming part of everyday applications across many industries.

![Amazon A+ 01](a-plus-01.png)

This second edition has been rewritten from the ground up to reflect how dramatically the AI landscape has evolved in recent years. This edition explores practical generative AI systems on AWS in much greater depth. You’ll work with services and solutions such as Amazon Bedrock, SageMaker AI, Bedrock AgentCore, and Strands Agents while building RAG-powered applications, AI agents, and automated workflows. You’ll also explore newer topics that have become increasingly important, including LLM-as-a-judge evaluation techniques, end-to-end automation with SageMaker Pipelines, and advanced model fine-tuning and deployment using SageMaker AI.

## Why is now the right time for engineers to master generative AI and LLMOps on AWS?

The demand for AI-powered applications is growing faster than most technology shifts in the past decade. Now is the right time for engineers to explore building generative AI systems on AWS, as AI is becoming central to how modern applications are designed and built. Many engineers focus on AI engineering at the application level, while more advanced practitioners go deeper into machine learning engineering and LLMOps to understand how models are fine-tuned, deployed, evaluated, and managed in production.

![Amazon A+ 02](a-plus-02.png)

Mastering generative AI and LLMOps is not something that happens overnight. In fact, it may take decades of continuous learning and hands-on experience to develop deep expertise across the many disciplines involved. The good news is that you do not need to master everything before you start building. Every project, experiment, and deployment helps you develop the skills needed to work effectively with these technologies in real-world environments.

## What makes this book different from other MLOps or generative AI books available today?

This book takes you through a practical, end-to-end process of building and managing generative AI systems and workflows on AWS. This book focuses on building complete, production-ready systems that combine AI agents, RAG-powered systems, and MLOps workflows in a unified and practical way. Each chapter is designed to equip you with the knowledge and skills you need, reinforcing what you learned previously while introducing new concepts that help you build working systems from end to end.

![Amazon A+ 03](a-plus-03.png)

I also assume you may be learning these services for the first time, so I introduce concepts gradually before moving into more advanced patterns. As you go through each chapter, you will learn how to troubleshoot common problems, apply best practices, and address real-world production requirements. I wanted this book to feel like the kind of book I wish I had when I was just getting started with AI, machine learning, and machine learning engineering. Back then, I often had to piece things together from scattered resources, and it was not always clear where to begin. This book is meant to give you that starting point and make the learning path more straightforward for generative AI, machine learning engineering, and MLOps.

## What will readers be able to build or deploy in real-world environments after reading this book?

After reading this book, you will be able to build and deploy real-world generative AI systems on AWS from end to end. You will build AI agents using Amazon Bedrock AgentCore and Strands Agents that interact with large language models deployed in a managed inference endpoint, enabling them to reason and respond in real applications. You will also build RAG-powered systems using Amazon Bedrock Knowledge Bases and S3 Vectors, allowing your applications to answer questions grounded in your own data.

![Amazon A+ 04](a-plus-04.png)

Along the way, you will also learn how to work like a machine learning engineer in production. You will fine-tune and deploy models using Amazon SageMaker AI, automate LLMOps workflows with SageMaker Pipelines, and evaluate model behavior using LLM-as-a-judge techniques. By the end, you will be able to combine these building blocks to design and engineer more complex AI-powered systems that are ready for real-world production use.

## How do you guide readers in building a complete RAG-powered application from data to deployment on AWS?

In the second chapter of this book, I will walk you through step by step how to build a complete RAG-powered application on AWS. I will guide you the same way you might learn to build a miniature bridge on a beach. You will begin by understanding how each part works and gradually integrate these building blocks into something more complete and functional.

![Amazon A+ 05](a-plus-05.png)

You will start with a sample dataset and upload it to Amazon S3. From there, you will learn how Amazon Bedrock Knowledge Bases will ingest the data, generate vector embeddings, and store them in a managed vector store such as Amazon S3 Vectors or other supported vector databases to make the content searchable. You will then connect this knowledge layer to a RAG-powered AI agent which reasons over the retrieved context and generates responses.
