# SMS Alert Setup Guide

## Quick Setup for SMS Alerts

### 1. Get Twilio Credentials

1. **Sign up for Twilio**: Go to [twilio.com](https://twilio.com) and create a free account
2. **Get your credentials**:
   - Account SID (starts with "AC...")
   - Auth Token (32 characters)
   - Phone Number (your Twilio number)

### 2. Configure SMS Settings

Edit `configs/alert_config.yaml`:

```yaml
# Twilio SMS Configuration
sms:
  enabled: true
  provider: twilio
  account_sid: "YOUR_TWILIO_ACCOUNT_SID"
  auth_token: "YOUR_TWILIO_AUTH_TOKEN"
  from_number: "YOUR_TWILIO_PHONE_NUMBER"

# SMS Recipients
recipients:
  sms:
    - "+1234567890"   # Add your phone numbers here
    - "+0987654321"   # Add more recipients as needed
```

### 3. Test SMS System

```bash
# Test SMS connection
python test_sms_friend_9044235343.py

# Run live demo
python live_demo.py
```

### 4. Important Notes

- **Twilio Restriction**: You cannot send SMS to your own Twilio number
- **International SMS**: Works globally (costs ~$0.0075 per SMS)
- **Free Trial**: Twilio provides $15 free credit for testing
- **Production**: Upgrade to paid account for production use

### 5. Troubleshooting

**SMS not working?**
- Check Twilio credentials are correct
- Verify phone numbers are in international format (+1234567890)
- Ensure Twilio account has sufficient balance
- Check Twilio logs for delivery status

**Getting errors?**
- Make sure virtual environment is activated
- Install requirements: `pip install -r requirements.txt`
- Check config file syntax is correct

### 6. Production Deployment

For production use:
1. **Upgrade Twilio account** to paid plan
2. **Add phone number verification** for recipients
3. **Set up monitoring** for SMS delivery
4. **Configure backup channels** (email, Slack)
5. **Test thoroughly** before going live

## Support

Need help? Check the main README.md or create an issue on GitHub.
