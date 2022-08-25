<script>
	export let title = "Coordinate Regression for event-based vision";
	export let description = "Coordinate regression for event-based vision";
	export let url = "https://jegp.github.io/coordinate-regression";
	export let keywords = "event-based vision, spiking neural networks, machine learning";
	export let preloadFont = [];
</script>

<svelte:head>
	<title>{title}</title>
	<meta name="description" content={description} />
	<meta name="author" content="Jens E. Pedersen" />
	<meta name="news_keywords" content={keywords} />

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:site" content="https://jegp.github.io/coordinate-regression" />
	<meta name="twitter:creator" content="@jensegholm" />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={description} />
	<meta name="twitter:image:src" content="{url}/assets/social-twitter.jpg" />

	<meta name="robots" content="max-image-preview:large" />

	<link rel="canonical" href="{url}/" />
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/all.min.css" integrity="sha512-1sCRPdkRXhBV2PBLUdRb4tMg1w2YPf37qatUFeS7zlBy7jJI8Lf4VHwWfZZfpXtYSLy85pkm9GaYVYMfw5BC1A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
	
	{#each preloadFont as href}
		<link rel="preload" {href} as="font" type="font/woff2" crossorigin />
	{/each}
</svelte:head>
