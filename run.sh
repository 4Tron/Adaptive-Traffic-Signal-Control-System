fname=out.txt
if test -e $fname
then
	rm $fname
fi
python multithreading.py "/videos/1.mp4" >> out.txt
python multithreading.py "/videos/2.mp4" >> out.txt
python multithreading.py "/videos/3.mp4" >> out.txt
python multithreading.py "/videos/4.mp4" >> out.txt
python program.py
