from __future__ import annotations

from typing import Dict


_DEFAULT_TREATMENT = {
	"treatment": "Use a safe, eco-friendly bio-fungicide as per local agronomy guidance.",
	"prevention": "Maintain field hygiene, remove infected leaves, and avoid over-irrigation.",
}


_TREATMENT_MAP: Dict[str, Dict[str, str]] = {
	"blast": {
		"treatment": "Spray neem oil (5 ml/L) or Trichoderma-based bio-fungicide every 7 days.",
		"prevention": "Ensure proper drainage and avoid excess nitrogen fertilizer.",
	},
	"bacterial_blight": {
		"treatment": "Apply Pseudomonas fluorescens bio-agent at recommended intervals.",
		"prevention": "Use certified disease-free seeds and prevent water stagnation.",
	},
	"brown_spot": {
		"treatment": "Apply compost tea or neem-based fungicide every 8 days.",
		"prevention": "Maintain balanced fertilization and proper plant spacing.",
	},
	"tungro": {
		"treatment": "Control leafhoppers using neem-based insecticide.",
		"prevention": "Remove infected plants and control vector population early.",
	},
	"black_rust": {
		"treatment": "Apply Trichoderma bio-fungicide as per label dosage.",
		"prevention": "Remove infected debris and improve air circulation.",
	},
	"brown_rust": {
		"treatment": "Use Bacillus subtilis bio-fungicide.",
		"prevention": "Plant resistant varieties and monitor humidity.",
	},
	"yellow_rust": {
		"treatment": "Spray neem oil (4 ml/L) at 7-day intervals.",
		"prevention": "Avoid dense planting and promptly remove infected leaves.",
	},
	"cercospora_leaf_spot": {
		"treatment": "Spray neem extract or copper-based bio-fungicide.",
		"prevention": "Use crop rotation and remove infected leaf material.",
	},
	"common_rust": {
		"treatment": "Apply Bacillus-based bio-fungicide at recommended intervals.",
		"prevention": "Ensure adequate spacing and maintain field sanitation.",
	},
	"northern_leaf_blight": {
		"treatment": "Use Trichoderma-based organic fungicide.",
		"prevention": "Avoid overhead irrigation and rotate crops each season.",
	},
	"healthy": {
		"treatment": "No treatment needed; continue standard eco-friendly crop care.",
		"prevention": "Monitor crop health regularly and maintain balanced nutrition.",
	},
}


def _normalize_disease_key(disease: str) -> str:
	return (disease or "").strip().lower().replace(" ", "_")


def get_treatment(disease: str) -> Dict[str, str]:
	"""Return eco-friendly treatment and prevention for a disease label."""
	return _TREATMENT_MAP.get(_normalize_disease_key(disease), _DEFAULT_TREATMENT)
