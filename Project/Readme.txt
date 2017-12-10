GRAPHICS PROGRAMMING PROJECT
JAVIER MANTILLA
G00329649
GMIT

This is my submission for the Graphics Programming module for the winter term of 2017.
This project is a game conisting of a hero that captures monsters.

Project folder tree structure:
This project consists in a file called 'project.html' where all the
game logic resides and a subdirectory called 'images' where the
background and sprites of the game are.

Game Objective:
The hero has to capture all the monsters in a level without getting caught by the devil.
When the player passes to the next level when catches all monsters.
This game has infinite levels.
This game ends when the hero is caught by the devil.

Game Instructions:
The player has to open the file 'project.html' in a HTML 5 compatible
web browser. 

Using the arrow keys the player controls the hero.
If the player is caught by the devil, a 'GAME OVER'
signs shows up indicating the end of the game.

The player can start a new session by clicking in the text:
'click here to restart'.

This project satisfies the following criteria:
- User interaction: The player uses the arrow keys to control the main game character.
- Basic collision detection: When the player capture a monster or is caught by the devil
        collision detection is used. Also when the monsters collide among them or against
        the walls.
- Cartesian to polar coordinates: When to monsters collide among them, a change of
        coordinate system is performed in order to know the new direction of the monsters.
- Illustration of movement: All characters move arround inside the canvas. This movement
        allows the game to be played.
- Multiple moving objects: The game keeps track of the characters movement and the collisions
        between them and their surroundings.
- Sprites: All game characters have representative sprites.
                