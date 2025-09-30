#!/bin/bash

# The name of your firewall rule
FIREWALL_RULE_NAME="allow-my-ip-on-8000"

# Get your current public IP address
CURRENT_IP=$(curl -s ifconfig.me)

# Check if an IP was retrieved
if [ -z "$CURRENT_IP" ]; then
  echo "Could not retrieve current IP address. Exiting."
  exit 1
fi

# Update the firewall rule with your new IP
gcloud compute firewall-rules update $FIREWALL_RULE_NAME \
    --source-ranges="$CURRENT_IP/32"

echo "Firewall rule '$FIREWALL_RULE_NAME' updated to allow traffic from your new IP: $CURRENT_IP"