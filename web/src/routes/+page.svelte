<script>
	import { setContext } from "svelte";
	import { browser } from "$app/env";
	import Meta from "$components/Meta.svelte";
	import Index from "$routes/index.svelte";
	import version from "$utils/version.js";

	export let data;

	version();

	const preloadFont = [
		"https://pudding.cool/assets/fonts/tiempos/TiemposTextWeb-Regular.woff2",
		"https://pudding.cool/assets/fonts/tiempos/TiemposTextWeb-Bold.woff2",
		"https://pudding.cool/assets/fonts/national/National2Web-Regular.woff2",
		"https://pudding.cool/assets/fonts/national/National2Web-Bold.woff2"
	];

</script>

<Meta />
<Index />
