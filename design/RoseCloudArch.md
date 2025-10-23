# Rose Application

Rose as a web application consists of 3 components:

* This repository, which contains the core algorithms and experimentation
* [Rose Server](https://github.com/AO-StreetArt/Rose-Server), which contains a backend for serving up a React application, and facilitating communication between that app and AWS Bedrock/Sagemaker services
* [Rose UI](https://github.com/AO-StreetArt/Rose-UI), which contains a React app that exposes a chat window, 2d image viewer, and 3d viewer.

## AWS Cloud Native

Our architecture overall consists of:

* An AWS Bedrock Agent, configured with a variety of Action Groups
* A variety of AWS Sagemaker endpoints (to be transitioned to Lambdas to minimize cost)
* Simple AWS Lambdas configured in Agent Action Groups for functions like Web Search & Arithmetic
* Rose-Server, acting as primary web hosting platform
* Elastic ELB, as load-balancer for Rose-Server
* Route53 for accepting public traffic

### Security

* Internal Services are on a VPC, with Rose-Server being exposed publicly through ELB & Route53
* HTTPS encryption should be enabled, with OAuth2.0 & JWT flows for authentication

### Sagemaker Endpoints

We use one pre-built Sagemaker endpoint (Semantic Segmentation), and the model_builds/ folder in this repository contains custom model containers which can be exposed in Sagemaker.  However, running these is expensive - we should try running on Lambdas instead until we have lots of users to minimize cost.

## Terraform

AWS Infrastructure should be pulled down into Terraform files and kept there.