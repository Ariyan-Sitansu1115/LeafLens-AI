"""
Language Manager - Manages language loading and caching
"""

import json
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class LanguageManager:
    """Manages language files and translations"""
    
    def __init__(self):
        self.locales_path = os.path.join(os.path.dirname(__file__), 'locales')
        self.supported_languages = {
            'en': 'English',
            'hi': 'हिंदी (Hindi)',
            'od': 'ଓଡ଼ିଆ (Odia)',
            'ta': 'தமிழ் (Tamil)',
            'te': 'తెలుగు (Telugu)',
            'bn': 'বাংলা (Bengali)',
            'gu': 'ગુજરાતી (Gujarati)',
            'mr': 'मराठी (Marathi)',
            'kn': 'ಕನ್ನಡ (Kannada)',
            'ml': 'മലയാളം (Malayalam)'
        }
        self.language_code_to_llm_name = {
            'en': 'English',
            'hi': 'Hindi',
            'od': 'Odia',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'kn': 'Kannada',
            'ml': 'Malayalam',
        }
        self.translations = {}
        self.default_language = 'en'
        self._load_all_languages()
        logger.info(f"✓ Language Manager initialized with {len(self.translations)} languages")
    
    def _load_all_languages(self):
        """Load all language files"""
        for lang_code in self.supported_languages.keys():
            self._load_language(lang_code)
    
    def _load_language(self, language_code: str):
        """Load a specific language file"""
        file_path = os.path.join(self.locales_path, f'{language_code}.json')
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[language_code] = json.load(f)
                logger.info(f"✓ Loaded language: {language_code}")
            else:
                logger.warning(f"⚠️ Language file not found: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading language {language_code}: {str(e)}")
    
    def get_language(self, language_code: str) -> Optional[Dict]:
        """Get language translations"""
        language_code = language_code.lower()
        
        if language_code in self.translations:
            return self.translations[language_code]
        
        logger.warning(f"⚠️ Language {language_code} not found, using default {self.default_language}")
        return self.translations.get(self.default_language, {})
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.supported_languages

    def get_supported_languages_detailed(self) -> Dict[str, Dict[str, str]]:
        """Get supported languages with both UI and LLM names."""
        details: Dict[str, Dict[str, str]] = {}
        for code, display_name in self.supported_languages.items():
            details[code] = {
                "display_name": display_name,
                "llm_name": self.language_code_to_llm_name.get(code, "English"),
            }
        return details
    
    def translate(self, language_code: str, key: str, default: str = None) -> str:
        """Get translation for a key"""
        translations = self.get_language(language_code)
        
        # Handle nested keys (e.g., "weather_display.temperature")
        keys = key.split('.')
        value = translations
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default or key
        
        return value if value else (default or key)
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code.lower() in self.supported_languages

    def resolve_language_code(self, language: str) -> str:
        """Resolve language code from either code or language name.

        Accepts values like "en", "English", "हिंदी", or "हिंदी (Hindi)".
        Returns default language if value is not supported.
        """
        if not language:
            return self.default_language

        value = language.strip().lower()
        if value in self.supported_languages:
            return value

        # Match against "English" part in "हिंदी (Hindi)" style labels
        for code, display_name in self.supported_languages.items():
            display_lower = display_name.lower()
            if value == display_lower:
                return code

            if "(" in display_name and ")" in display_name:
                english_alias = display_name.split("(", 1)[1].split(")", 1)[0].strip().lower()
                if value == english_alias:
                    return code

            # Match native prefix (before bracket), e.g. "हिंदी"
            native_alias = display_name.split("(", 1)[0].strip().lower()
            if value == native_alias:
                return code

        logger.warning("⚠️ Could not resolve language '%s'; using default %s", language, self.default_language)
        return self.default_language

    def resolve_language(self, language: str) -> Tuple[str, str]:
        """Resolve to `(language_code, llm_language_name)`."""
        code = self.resolve_language_code(language)
        return code, self.language_code_to_llm_name.get(code, "English")