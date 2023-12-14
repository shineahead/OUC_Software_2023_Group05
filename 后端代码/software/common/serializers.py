from rest_framework import serializers

class SARInfo(serializers.ModelSerializer):
    def validate_content(self, value):
        return value

