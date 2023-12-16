
**Columbia MS in Data Science Capstone project : Repository for analysis of images taken over Greenland**

# Introduction
The Greenland ice sheet (GrIS), one of the world's largest ice masses, plays a critical role in Earth's climate system and sea level dynamics. Recent decades have seen accelerated melting and changes in the ice sheet's albedo, which is the fraction of incoming solar radiation that is reflected from the ice surface. Albedo plays a pivotal role in regulating energy balance and meltwater production, thereby contributing to sea level rise. In recent years, the decreasing albedo over the GrIS has raised concerns about its future stability and contributions to sea-level rise. This research explores the background of the GrIS and the objectives of an expedition aimed at understanding its changing dynamics. 

Unique multi-spectral and multi-altitude images were captured in mid 2023 which form the basis of this analysis. In particular, this code focuses on using computer vision on images taken via drone to classify portions of land in the context of unsupervised machine learning in glacial science. We seek to identify and understand surface characteristics like ice, glacier, and vegetation spots, and also establish connections between spectral properties and albedo. The subsequent step to this is to evaluate the information gain or loss across various elevations using the same techniques. 


# Using this Repo
### Required Packages
The python packages required to run the notebooks are included in the requirements.yml file.
The Python packages required to run the W-Net architecture are included in w_net_environment.yml.

### Underlying Data
Data (images of Greenland) are hosted on Google Drive and partially replicated locally for analysis. Raw images were processed with Pix4D Mapper to correct for albedo adjustments and to stich together into mosaics. If you would like access to the data, please reach out. 

### Example Image
![alt text](https://github.com/ds4114/Greenland_Drone_Analysis/blob/main/docs/DJI_20230715082014_0010_D.jpg)


# Repo Contents
TBD description of code files

The data folder structure is as follows:
  - ./data/pixel_mosaics: contains 3 sub folders by altitude with raw uncorrected images (one image per the 4 channels)
  - ./datareflectance_mosaics: contains 3 sub folders by altitude with reflectance corrected images (one image per the 4 channels)
  - ./data/reflectance_matching: contains 3 sub folders by altitude with cropped corrected images showing only overlapping land area between altitudes (one image per the 4 channels)


# Results and Takeaways
- Paper:  https://docs.google.com/document/d/1ozPPcIoGJ62VficfdIoqLq4zrUzDLEBl/edit
- Poster: https://docs.google.com/presentation/d/1-VtdJFuupKcrpl2XHIIQlF2CuT93zejB/edit

# Acknowledgements
Sponsor: Marco Tedesco, Columbia Lamont-Doherty Earth Observatory

Team Members (Alphabetically): 
Braden Jacob Huffman (bjh2179) , Devan Samant (ds4114) ,Hugo Ginoux (hg2632), Samuel Edward Fields (sef2186) ,Sushant Surendra Prabhu (ssp2202)

