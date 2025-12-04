# Image Enhancement for Color Vision Deficiency based on Empirical Color Naming Simulation

## Background  
CVD (Color Vision Deficiency) can cause significant challenges in visual communication and interaction, as it can make it difficult for individuals with CVD to distinguish between colors and follow color instructions. While existing CVD color filter algorithms address the seperation of color components for CVD (e.g. color filter in iOS and Android for accessibility), our color filter also make the color namable for CVD. In this case, CVD can distinguish and identify the color correctly. 

First, we need to learn how CVD perceive and identify the color. The input light is transformerd into LMS channel signal by cone cells. 
For CVD, L/M channel output is close to each other, causing dimension reduction. However, their color naming abality is learned from color normals, which means they are always trying to identify color 

## TODO List

-[x] 1. Filter Model    
-[x] 2. Degradation Model (CVD simulator with torch framework)    
-[x] 3. DIY SFT Trainer (Train on color words only)    
-[o] 4. DIY Processor  
    -[x] * Design a data collator. Make sure the padding works  
    -[x] * Recover the image from the image token.   
    ~~-[] * Let a token based filter to process image. Then recover the image. ~~  
-[] 4. Pipeline (Join models, and Training script)    
    -[x] * Freeze Parts  
    -[x] * Concat model with proper dtype and device  
    -[] * Try the pipeline  
    -[x] * Add multitarget loss  
    
DEBUG：
-[] Data-Trainer
-[] Processor
-[] Inference (Model)
-[] Batch Inference
-[] Trainer
  
## Discussion
* 用哪个框架？
* ms-swift.     好处：数据集格式处理比较省心，加速训练      坏处：魔改模型后不知道还能不能正常工作（包括template、tokenizer等），以及data collator实现细节不清晰
[*] TRL.          好处：细节透明，处理起来有经验，扩展性强    坏处：数据集格式可能要改，可能不支持qwen3 vl

折中方案：只用Swift的trainer，以及dataset generator。其他实现用TRL标准