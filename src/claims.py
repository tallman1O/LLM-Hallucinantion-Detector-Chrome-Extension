from src.claim_extractor import extract_verifiable_claims


def split_into_claims(text: str):
    claims, _skipped = extract_verifiable_claims(text)
    return claims
