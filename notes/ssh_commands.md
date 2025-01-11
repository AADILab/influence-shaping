To close a frozen ssh session

Press the following keys
1. Enter
2. ~ (tilde)
3. . (period)

To send a keep-alive to keep ssh from timing out
ssh -o ServerAliveInterval=60 myname@myhost.com
