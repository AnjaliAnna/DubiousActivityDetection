

predictions = model.predict_classes(test_image)

# print("The screen time of POTENTIAL is", predictions[predictions==1].shape[0], "seconds")
# print("The screen time of HARM  is", predictions[predictions==2].shape[0], "seconds")
# print("The screen time of SAFE is", predictions[predictions==0].shape[0], "seconds")

harm = predictions[predictions == 2].shape[0]
pot = predictions[predictions == 1].shape[0]
safe = predictions[predictions == 0].shape[0]

#   (pot >= harm) and

if (harm >= pot) and (harm >= safe):
    print("The video contains Dubious Activity")
elif (pot >= harm) and (pot >= safe):
    print("The video contains Potentially Dubious Activity")
else:
    print("The video contains No Dubious Activity. The video is safe")