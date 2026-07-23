# BISONN

## What is BISONN
BISONN or Biotic Interactions with Sage Observations using Neural Networks is a project to start using AI tools to classify biotic interactions from camera data. This project, in its current state, is a proof of concept example and performance comparison for models attempting to classify biotic interactions.


## Background
Biodiversity is essential for ecosystem functioning and the provisioning of ecosystem services but is increasingly threatened by global change. Global change is impacting species abundances, distributions, and biotic interactions causing novel ecological communities, and doing so at rates more rapidly than ever before 5,6. Understanding and mitigating these changes has therefore become critical to biodiversity conservation.

However, predicting biodiversity responses to global change remains challenging because the ecological processes that shape species populations and communities operate across multiple spatial and temporal scales. Species distributions and community composition emerge from interactions among processes ranging from local-scale habitat conditions and species interactions to landscape-scale environmental gradients and regional climate patterns. This is a central challenge in ecology, processes governing biodiversity are inherently scale dependent, yet most predictive models assume that relationships between biodiversity and other drivers remain constant across scales. While studies have modeled how geophysical and biological drivers (i.e., forest structure, topographic characteristics, landscape composition and configuration) influence biodiversity across multiple spatial and temporal scales, most approaches only emphasize coarse climate or landscape patterns and overlook other important drivers that shape ecological communities.

Perhaps the greatest remaining challenge in predictive ecology is incorporating biotic interactions into models of biodiversity change. Although ecological theory recognizes species interactions as fundamental determinants of biodiversity, they are rarely incorporated into models because they are difficult to quantify consistently across broad spatial and temporal extents. While studies have shown the importance of biotic interactions on species distributions and community composition, we don’t understand how these effects operate at different scales, i.e. the Raunkiaeran shortfall, especially in relation to other drivers. Similarly, we don’t know how biotic interactions relate to other critical components of biodiversity distribution like geophysical characteristics (e.g., climate, roughness, topographic complexity, etc.) which inherently mediate the importance of biological drivers like biotic interactions.

Here, we begin to explore the ability of AI to collect biotic interaction data by working with the Sage Grande Testbed and applying a simple example model at the edge. We compare two foundation models with a few different classification heads trained on the same data to perform the same task: classify an image to a certain biotic interaction type.



