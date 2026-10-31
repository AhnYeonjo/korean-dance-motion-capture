# 02 · Drone-Based Detection of Wandering (Research Proposal)

*Research plan; not carried out as an experiment. Written in Korean; summarized here in English.*

## Problem
People living with dementia or autism can wander away from home and go missing. Search often relies on how a person looks, which fails at a distance or in poor footage.

## Proposed approach
**A. Anomaly detection for wandering behavior**
- Data: AI Hub *Senior abnormal-behavior video* and *Abnormal-behavior CCTV video* (12 behavior classes including wandering and fainting, ~717 h), plus the UCI *Daily and Sports Activities* dataset.
- Model: a convolutional autoencoder trained on normal behavior. Footage it reconstructs poorly gets flagged as abnormal. The plan compares supervised and semi-supervised anomaly detection.
- Experiment: train on different mixtures of the datasets and compare detection accuracy.
- Use: send detections and live location to police through an API so they can respond quickly.

**B. Finding a specific person by gait**
- Recognize a missing person from how they walk when the face or clothing can't be seen.
- Datasets: GREW, Gait3D, GaitLU-1M, SUSTech1K, DroneGait (aerial view). Framework: OpenGait with the BigGait model.
- Plan: fine-tune on a short clip of one person, then test whether a drone can find that person around the school.

## What it led to
The proposal treats **movement as a signal**: abnormal behavior and individual identity can both be read from how a body moves. That idea carries into [03 · gait analysis for early Alzheimer's detection](../03-gait-alzheimers/).
