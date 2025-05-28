# Environment Variables

To work with remote datasets or accessing resources on remote paths, the CLI looks up the required authentication data from environment variables.

## Webknossos Datasets

When working with non-public webknossos datasets or an own webknossos instance, these variables can be used:

- **WK_TOKEN**  
    _This is the auth token to identify against webknossos. It can be generated [here](https://webknossos.org/auth/token)._
- **WK_URL**  
    _This is the base URL for webknossos APIs and datasets. If it is unset the default ***https://webknossos.org*** is used._

## AWS S3 Configuration

Use these variables when accessing datasets stored on AWS S3 or compatible services:

- **AWS_ACCESS_KEY_ID**  
    _Your AWS access key ID used for authenticating with S3 services._
- **AWS_SECRET_ACCESS_KEY**  
    _Your AWS secret access key required for S3 authentication._
- **S3_ENDPOINT_URL**  
    _The endpoint URL for the S3 service; useful for S3-compatible storage providers._

## HTTP Basic Authentication

Use these variables when HTTP basic authentication is required:

- **HTTP_BASIC_USER**  
    _Username for HTTP basic authentication._
- **HTTP_BASIC_PASSWORD**  
    _Password associated with the HTTP basic username._