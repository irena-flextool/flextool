"""Solar System object names for generating node location names."""

# Curated from https://en.wikipedia.org/wiki/List_of_Solar_System_objects
# Planets, dwarf planets, major moons, and notable asteroids
SOLAR_SYSTEM_NAMES: list[str] = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Pluto", "Ceres", "Eris", "Haumea", "Makemake", "Sedna", "Quaoar", "Orcus",
    "Luna", "Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto",
    "Titan", "Rhea", "Iapetus", "Dione", "Tethys", "Enceladus", "Mimas",
    "Hyperion", "Phoebe", "Janus", "Epimetheus",
    "Titania", "Oberon", "Ariel", "Umbriel", "Miranda",
    "Triton", "Proteus", "Nereid",
    "Charon", "Nix", "Hydra", "Kerberos", "Styx",
    "Vesta", "Pallas", "Hygiea", "Juno", "Psyche", "Eros", "Ida", "Gaspra",
    "Bennu", "Ryugu", "Itokawa", "Mathilde", "Lutetia", "Steins",
    "Halley", "Hale", "Churyumov", "Borrelly", "Tempel", "Wild",
    "Varuna", "Ixion", "Huya", "Chaos", "Rhadamanthus",
    "Gonggong", "Salacia", "Varda", "Altjira", "Borasisi",
    "Dysnomia", "Namaka", "Hiiaka", "Vanth", "Weywot",
    "Arrokoth", "Albion", "Logos", "Typhon", "Lempo",
    "Deucalion", "Crantor", "Thereus", "Echeclus", "Bienor",
    "Chiron", "Pholus", "Nessus", "Chariklo", "Okyrhoe",
    "Elara", "Himalia", "Lysithea", "Carme", "Sinope",
    "Amalthea", "Thebe", "Adrastea", "Metis",
]


def get_location_names(count: int) -> list[str]:
    """Return a list of unique location names from the Solar System catalog."""
    if count > len(SOLAR_SYSTEM_NAMES):
        raise ValueError(
            f"Requested {count} names but only {len(SOLAR_SYSTEM_NAMES)} available"
        )
    return SOLAR_SYSTEM_NAMES[:count]
