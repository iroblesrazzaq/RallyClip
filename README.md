Tennis Tracker: Free, open source project to automate cutting dead time between points for easier point review. 


the problem with this approach is that the court itself (bounded by lines) isnt the potential playing area - the playing area is defined as the entire space where players can be during a point. so we'd have to annotate and train a model? or we could using yolo bounding boxes on our annotated matches to draw this, using only during point play? then we can train a CNN type model to output the area somehow (whats the best way to do this? some type of polygon?)

the problem with this approach is that 1. our yolo right now identifies players on adjacent court, and 2. places where the player doesn't go still are playable court, but just werent visited during that match.
