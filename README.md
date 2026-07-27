# BISONN
## Biotic Interactions with Sage Observations using Neural Networks (BISONN)

<br/>

A Sage/Waggle edge plugin that detects **bird mobbing behavior** and identifies
**species** from camera images using BioCLIP 2.5 embeddings. Behavior is
classified by a frozen Linear SVM (98.5% accuracy, 0.935 macro-F1). Species
is identified zero-shot via cosine similarity to 30 curated North American
bird text prompts — a task BioCLIP was explicitly trained for on TreeOfLife-200M.

<br/>

## Initial Sage Grande Testbed (SGT) Workshop Goals
- Develop a proof of concept example for monitoring biotic interactions at the edge using AI
- Build and test simple classificaiton heads for BioCLIP to quantify biotic interactions
- Compare performance of these classificaiton heads and of the base BioCLIP embeddings for quantifying biotic interactions
- Do the same for DINOv3 and compare performance among all models
- Publish the best model to the SGT at the edge

<br/>

## Long Term Goals
- Integrate multiple sensors (ARU, video, image) to monitor for interactions more effectively
- Develop a more powerful, well trained version of this model for many more types of biotic interactions
- Scale data collection across the SGT network
