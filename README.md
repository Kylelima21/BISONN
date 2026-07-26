# BISONN
## Biotic Interactions with Sage Observations using Neural Networks (BISONN)

<br/>

## Initial Sage Grande Testbed (SGT) Workshop Goals
- Develop a proof of concept example for monitoring biotic interactions at the edge using AI
- Build and test simple classificaiton heads for BioCLIP to quantify biotic interactions
- Compare performance of these classificaiton heads and of the base BioCLIP embeddings for quantifying biotic interactions
- Do the same for DINOv3 and compare performance among all models
- Publish the best model to the SGT at the edge

## Sage Plugin (Phase 4-5)

The deployable Sage/Waggle plugin lives at the repo root:

```
app.py              # Plugin entry point: camera → BioCLIP → SVM → publish
Dockerfile          # NVIDIA PyTorch 25.08 base (Blackwell sm_110), frozen torch
sage.yaml           # Sage ECR metadata (name=bisonn, version=0.1.0)
requirements.txt    # Plugin Python deps (torch frozen from base image)
models/svm.joblib   # Trained Linear SVM classifier (1.7 MB)
ecr-meta/           # Science description + icon/image for ECR portal
```

Build and test on a Thor node:
```bash
sudo pluginctl build .
sudo pluginctl run --name bisonn-test --selector zone=core \
  --resource limit.memory=16Gi,request.memory=4Gi \
  <image-ref> -- --image-dir /app/test-images --continuous N
```


<br/>

## Long Term Goals
- Integrate multiple sensors (ARU, video, image) to monitor for interactions more effectively
- Develop a more powerful, well trained version of this model for many more types of biotic interactions
- Scale data collection across the SGT network
