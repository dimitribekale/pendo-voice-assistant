import requests
import logging
import re
from src.config import Config
from .base_tool import BaseTool

# Configure logging
logger = logging.getLogger(__name__)

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__("Weather Tool", "Provides current weather information.")
        self.api_key = Config.OPENWEATHERMAP_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def _get_current_location(self):
        try:
            response = requests.get("https://ip-api.com/json/", timeout=5)
            response.raise_for_status()
            location_data = response.json()
            return location_data.get("city")
        except requests.exceptions.RequestException:
            return None

    def _validate_city_name(self, city):
        """
        Validates the city name input.

        Args:
            city: The city name string

        Returns:
            tuple: (is_valid, error_message, sanitized_city)
        """
        if not city:
            return False, "Please provide a city name.", None

        if not city.strip():
            return False, "City name cannot be empty.", None

        # Sanitize: strip whitespace
        city = city.strip()

        if len(city) > 100:
            return False, "City name is too long (max 100 characters).", None

        if len(city) < 2:
            return False, "City name is too short (min 2 characters).", None

        # Allow only letters, spaces, hyphens, and apostrophes (for cities like "New York" or "Saint-Denis")
        if not re.match(r"^[a-zA-Z\s\-']+$", city):
            return False, "City name contains invalid characters. Please use only letters, spaces, and hyphens.", None

        return True, "", city

    def _get_weather_data(self, city):
        if not self.api_key:
            return "API key for OpenWeatherMap is not provided. Please get a key from https://openweathermap.openweathermap.org/api"

        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"  # For Celsius
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
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
        """
        Gets weather information with input validation.

        Args:
            city: City name (optional, will use current location if not provided)

        Returns:
            str: Weather information or error message
        """
        if city:
            # Validate the provided city name
            is_valid, error_msg, sanitized_city = self._validate_city_name(city)
            if not is_valid:
                logger.warning(f"Invalid city name: {error_msg}")
                return error_msg

            logger.info(f"Getting weather for: '{sanitized_city}'")
            return self._get_weather_data(sanitized_city)
        else:
            # Try to get current location
            city = self._get_current_location()
            if city:
                logger.info(f"Using current location: '{city}'")
                return self._get_weather_data(city)
            else:
                logger.warning("Could not determine current location")
                return "I couldn't determine your current location."

