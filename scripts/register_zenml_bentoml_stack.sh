#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${ZENML_BENTOML_STACK_NAME:-local_bentoml_stack}"
DEPLOYER_NAME="${ZENML_BENTOML_DEPLOYER_NAME:-bentoml_deployer}"

zenml integration install bentoml -y

if ! zenml model-deployer describe "${DEPLOYER_NAME}" >/dev/null 2>&1; then
  zenml model-deployer register "${DEPLOYER_NAME}" --flavor=bentoml
fi

if ! zenml stack describe "${STACK_NAME}" >/dev/null 2>&1; then
  zenml stack register "${STACK_NAME}" \
    -o default \
    -a default \
    -d "${DEPLOYER_NAME}"
fi

zenml stack set "${STACK_NAME}"
