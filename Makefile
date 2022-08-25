PHONY: github netlify pudding

github:
	cd web && npm run build
	rm -rf docs
	cp -r web/build docs
	touch docs/.nojekyll
	
# aws-sync:
# 	aws s3 sync build s3://pudding.cool/year/month/name --delete --cache-control 'max-age=31536000'

# aws-cache:
# 	aws cloudfront create-invalidation --distribution-id E13X38CRR4E04D --paths '/year/month/name*'	

# pudding: aws-sync aws-cache
