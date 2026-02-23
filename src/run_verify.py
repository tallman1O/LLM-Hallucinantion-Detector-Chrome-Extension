import json
from verify import HallucinationVerifier

text = """
GANs completely fail on high-resolution images.
"""

verifier = HallucinationVerifier()
output = verifier.verify(text)

print(json.dumps(output, indent=2))