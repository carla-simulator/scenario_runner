from tkinter import *


import carla
import argparse

# Version of scenario_runner
VERSION = 1.0


DESCRIPTION = ("CARLA Weather Control: Change the weather of CARLA simulations\n" "Current version: " + str(VERSION))

PARSER = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
PARSER.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: localhost)')
PARSER.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')

ARGUMENTS = PARSER.parse_args()


_HOST_ = ARGUMENTS.host
_PORT_ = int(ARGUMENTS.port)
_SLEEP_TIME_ = 1


client = carla.Client(_HOST_, _PORT_)
client.set_timeout(2.0)
world = client.get_world()

weather = world.get_weather()

curr_cloudiness 	= weather.cloudyness
curr_precipitation 	= weather.precipitation
curr_precipitation_deposits 	= weather.precipitation_deposits
curr_wind_intensity = weather.wind_intensity
curr_sun_azimuth 	= weather.sun_azimuth_angle
curr_sun_altitude 	= weather.sun_altitude_angle

good_cloudiness 	= 0
good_precipitation 	= 0
good_precipitation_deposits 	= 0
good_wind_intensity = 0
good_sun_azimuth 	= 360
good_sun_altitude 	= 90



def set_weather(val = None):
	curr_cloudiness 			= w0.get()
	curr_precipitation 			= w1.get()
	curr_precipitation_deposits = w2.get()
	curr_wind_intensity 		= w3.get()
	curr_sun_azimuth 			= w4.get()
	curr_sun_altitude 			= w5.get()
	weather = carla.WeatherParameters(
		cloudyness 				= curr_cloudiness,
		precipitation 			= curr_precipitation,
		precipitation_deposits 	= curr_precipitation_deposits,
		wind_intensity 			= curr_wind_intensity,
		sun_azimuth_angle 		= curr_sun_azimuth,
		sun_altitude_angle 		= curr_sun_altitude 
		)
	world.set_weather(weather)

def good_weather(val = None):
	curr_cloudiness 			= good_cloudiness
	curr_precipitation 			= good_precipitation
	curr_precipitation_deposits = good_precipitation_deposits
	curr_wind_intensity 		= good_wind_intensity
	curr_sun_azimuth 			= good_sun_azimuth
	curr_sun_altitude 			= good_sun_altitude
	weather = carla.WeatherParameters(
		cloudyness 				= curr_cloudiness,
		precipitation 			= curr_precipitation,
		precipitation_deposits 	= curr_precipitation_deposits,
		wind_intensity 			= curr_wind_intensity,
		sun_azimuth_angle 		= curr_sun_azimuth,
		sun_altitude_angle 		= curr_sun_altitude 
		)
	world.set_weather(weather)
	w0.set(curr_cloudiness)
	w1.set(curr_precipitation)
	w2.set(curr_precipitation_deposits)
	w3.set(curr_wind_intensity)
	w4.set(curr_sun_azimuth)
	w5.set(curr_sun_altitude)	

window = Tk()
window.title("Change the weather in the Carla world!")

# creating 3 text labels and input labels
Label(window, text = "Cloudyness")				.grid(row = 0) # this is placed in 0 0
Label(window, text = "Precipitation")			.grid(row = 1) # this is placed in 1 0
Label(window, text = "Precipitation deposit")	.grid(row = 2) # this is placed in 2 0
Label(window, text = "Wind intensity")			.grid(row = 3) # this is placed in 3 0
Label(window, text = "Sun azimuth angle")		.grid(row = 4) # this is placed in 4 0
Label(window, text = "Sun altitude angle")		.grid(row = 5) # this is placed in 5 0

w0 = Scale(window, from_=0, to=100, tickinterval=10, orient=HORIZONTAL, length=600, command=set_weather)
w1 = Scale(window, from_=0, to=100, tickinterval=10, orient=HORIZONTAL, length=600, command=set_weather)
w2 = Scale(window, from_=0, to=100, tickinterval=10, orient=HORIZONTAL, length=600, command=set_weather)
w3 = Scale(window, from_=0, to=100, tickinterval=10, orient=HORIZONTAL, length=600, command=set_weather)
w4 = Scale(window, from_=0, to=360, tickinterval=45, orient=HORIZONTAL, length=600, command=set_weather)
w5 = Scale(window, from_=-90, to=90,tickinterval=10, orient=HORIZONTAL, length=600, command=set_weather)

w0.grid(row = 0, column = 1) # this is placed in 0 1
w1.grid(row = 1, column = 1) # this is placed in 1 1
w2.grid(row = 2, column = 1) # this is placed in 2 1
w3.grid(row = 3, column = 1) # this is placed in 3 1
w4.grid(row = 4, column = 1) # this is placed in 4 1
w5.grid(row = 5, column = 1) # this is placed in 5 1

w0.set(curr_cloudiness)
w1.set(curr_precipitation)
w2.set(curr_precipitation_deposits)
w3.set(curr_wind_intensity)
w4.set(curr_sun_azimuth)
w5.set(curr_sun_altitude)


# 'Checkbutton' is used to create the check buttons
Button(window, text = "Set good weather!", command=good_weather).grid(columnspan = 2) # 'columnspan' tells to take the width of 2 columns


mainloop()