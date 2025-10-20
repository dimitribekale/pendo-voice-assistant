import requests
from src.config import Config
from .base_tool import BaseTool

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__("Weather Tool", "Provides current weather information.")
        self.api_key = Config.OPENWEATHERMAP_API_KEY
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def _get_current_location(self):
        try:
            response = requests.get("http://ip-api.com/json/")
            response.raise_for_status()
            location_data = response.json()
            return location_data.get("city")
        except requests.exceptions.RequestException:
            return None

    def _get_weather_data(self, city):
        if not self.api_key:
            return "API key for OpenWeatherMap is not provided. Please get a key from https://openweathermap.openweathermap.org/api"

        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"  # For Celsius
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            weather_data = response.json()

            if weather_data["cod"] != 200:
                return f"Could not get weather for {city}. Error: {weather_data.get('message', 'Unknown error')}"

            main_weather = weather_data["weather"][0]["main"]
            description = weather_data["weather"][0]["description"]
            temperature = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]

            return f"The weather in {city} is {main_weather} ({description}). The temperature is {temperature}°C with {humidity}% humidity."

        except requests.exceptions.RequestException as e:
            return f"Could not get weather for {city}. Error: {e}"

    def run(self, city=None):
        if city:
            return self._get_weather_data(city)
        else:
            city = self._get_current_location()
            if city:
                return self._get_weather_data(city)
            else:
                return "I couldn't determine your current location."

