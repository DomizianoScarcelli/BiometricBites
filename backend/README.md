All backend files are located in backend/bsprojectbackend folder (the structure may change when we start working on the project).
Do not delete manage.py file!
You may want to create a virtual environment before starting working on the project.

If you're on Windows, type this command:
python -m venv "path to the BS-PROJECT/backend" folder
You may need to change "python" with the name of python's enviromental variable on your pc (often "python3" or "py"). After that, put yourself in the backend folder using "cd backend" and start the virtual environment typing "./Script\activate".
Install Django by typing "pip install Django" (assuming that you use pip).

Different commands might be needed for Linux/Mac OS. Look these sources for clarification:
https://code.visualstudio.com/docs/python/tutorial-django
https://www.youtube.com/watch?v=Wfu5dPbiyKA&list=WL&index=2&ab_channel=CodingEntrepreneurs

You can also change the interpreter in order to use the one of the virtual machine. Go to:
View -> Command Palette -> Python: Select Interpreter
And here select the interpreter of the VM (it should be something like "Python 3.9 ('backend: venv')").

Don't add lib files to Github, just use .gitignore file to avoid it.

Use "python manage.py runserver" to run the server.