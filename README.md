# Assignment

My attempt at tackling some of the challenges stated in the  take-home case study.


## Introduction

This README is the accompagnying deliverable document to the given case study. It describes the work conducted for this take-home data science assignment.    
  
Its main sections match the measures of improvements steps that are stated in the case study - these have driven the work. 
To showcase working mechanics, I choose a github repository for code, and github notes/issues for tasks break-down.  
  
Codes, reports and figures can be found in [here](https://github.com/marcalph/assignment).  
  
> While this document is hopefully not hard to read, I strongly advise the reader to acquaint him/(her)self with these sections through the corresponding project board, [here](https://github.com/marcalph/assignment/projects/1).  
The cards offer a more practical overview. A simple Kaban was used.

### Disclaimer

- while data quality was risen as a possible issue, I essentially consider data to be holding the truth (even regarding dramatic age range for example);  
- doing this case study, I came to realise in the recent years (2 or so) I spent a growing time doing deep learning over "traditionnal" data science, so I spent a fairly large amount on the case study because of time spent recoding snippets that were not available in my collection anymore...; 
- some tasks may appear closed while not completed or vice-versa because of a lack of understanding of what constitute the definition of done; 



## Repository checks and description


#### Repo checks
This repository follows my usual code layout for a work project.  
As a mean to enforce good data science practices, I have tried to include traditionnal software engineering tools :
- an src layout;
- unit tests;
- code linting & formatting;
- ~~typing~~*;
- a working demo.

\* well...ok maybe time felt short at the end. 
  
#### Repo description
  
The main folders of this repository are:
- assets \> basically all the static files;
- demo \> the demo contianer configuration files;
- src \> the actual code splitted into scripts;
- test \> minimal tests;


## Baseline mock

The first task I tackled (or rather attempted to) was to replicate the model described as a baseline in the case study. Several reasons motivated this choice:

- The model is "simple";  
   For the record, it is described as a logistic regression model with an elastic net penalty (i.e. L1 et L2 regularization) thus easy to train on a limited computational budget.
- Some of the data at hand has already been cleaned;
   This allows to iterate quickly without spending too much time on a thorough data analysis.
- It is a habit :bowtie:;
- It allows to develop a minimal end-to-end pipeline showcasing all the usual/essential steps of a ML project and gives solid grounds to build things further up.

The chosen pipeline, is described hereafter : 
1. EDA (exploratory data analysis)
2. Data processing
3. Model training
4. Model review
5. Model testing 
6. Exposition (which is viewed as mean to end-to-end testing and UATs)


A detailed description of this section can be found [here](assets/output/working_doc/baseline_mock.md)


## Model Interpretability
  
The idea of a interpretable model has somewhat always eluded me, as even linear model tend to be dramatically hard to understand depending on the features or preprocessing at hand. I've had the chance (sic.) to analyse models with over 200 features...
  
That being said model interpretability is almost systematically a key feature of a project - as however well performing a model can be, its perceived lack of transparency by the users can prevent its adoption.  
   
Common tools and frameworks, for this task are graphical ones.   
  
Be it, partial dependency plots, individual conditional expectations, `shap`, `lime` and `captum`.  
  
If captum is usually my prefered choice, is it almost exclusively used in a deep learning context (a `pytorch` one at that).  
Conceptually, Shapley values seems to be the most elegant solution to interpretabilityand the most solid technically speaking, but I find `lime` easier to explain and computationaly more efficient.  
For this reason, it is the main approach taken here and is avalaible as a dedicated endpoint in the accompanying demo container/API.  
  
Nevertheless, both approach are used in this case study, and codes are available at **src/utils/interpret.py**. 

An example of the interpretabilty tool is  available in the reports [subfolder](assets/output/reports).



## Feature engineering

lorem ipsum

## Model /Model Architecture


class sampling ?

but averageP seems better alternative (auc consider tpr/fpr avgP considerts tpr and precision)
todo if LR model optimize directly in chosen DL framework
metric tied to business considerations (order importance top 100 but first half or last?, workforce size w.r.t. lead transformation capacity (10,100,1000) metric@k, graded relevance? given target) so no choice a priori but definitely IR related AvgP/DCG


lorem ipsum







## Others


## Demo / Staging environment

lorem ipsum

## Automation

lorem ipsum

## Monitoring

lorem ipsum





