#!/bin/bash

# Script to set up passwordless cache dropping
# Run this script with sudo

echo "Setting up passwordless cache dropping..."

# Create the drop-caches script
cat > /usr/local/bin/drop-caches << 'EOF'
#!/bin/bash
# Drop page cache, dentries and inodes
sync && echo 3 > /proc/sys/vm/drop_caches
EOF

# Make it executable
chmod +x /usr/local/bin/drop-caches

# Get the current username
USERNAME=$(logname)

# Add sudoers entry for passwordless execution
echo "Adding sudoers entry for user: $USERNAME"
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/local/bin/drop-caches" >> /etc/sudoers.d/drop-caches

# Set proper permissions on sudoers file
chmod 440 /etc/sudoers.d/drop-caches

echo "Setup complete!"
echo ""
echo "You can now use /usr/local/bin/drop-caches without a password."
echo "Test it by running: /usr/local/bin/drop-caches"
echo ""
echo "Note: If you want to allow direct execution without sudo, you would need to:"
echo "1. Make the script setuid root (security risk!), or"
echo "2. Use a capability-based approach (complex), or"
echo "3. Always call it via: sudo /usr/local/bin/drop-caches (no password needed)"