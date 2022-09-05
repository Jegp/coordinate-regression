<script>
	const links = [
		{
			name: "about",
			url: "https://jepedersen.dk/about",
			icon: "fa-solid fa-house"
		},
		{
			name: "twitter",
			url: "https://twitter.com/jensegholm/",
			icon: "fab fa-twitter"
		},
		{
			name: "GitHub",
			url: "https://github.com/jegp/coordinate-regression/",
			icon: "fab fa-github"
		}
	];
</script>

<footer>
	<section class="about">
		<p>
			This website was developed by <a href="https://jepedersen.dk"
				>Jens Egholm Pedersen</a
			>
			who is studying at the
			<a href="https://neurocomputing.systems">Neurocomputing Systems lab</a>
			at the <a href="https://kth.se">KTH Royal Institute of Technology</a> in Stockholm,
			Sweden.
		</p>
		<p>The work is licensed under LGPLv3.</p>
	</section>

	<section class="links">
		<ul>
			{#each links as link}
				<li>
					<a href={link.url}>
						<span><i class={link.icon} /> {link.name.toUpperCase()}</span>
					</a>
				</li>
			{/each}
		</ul>
	</section>
</footer>

<style>
	footer {
		background-color: var(--color-fg);
		color: var(--color-bg);
		font-family: var(--sans);
		padding: 3em 1em;
		margin-top: 3em;
	}

	a,
	a:visited,
	a:hover {
		color: var(--color-bg);
	}

	.stories {
		margin: 0 auto;
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		max-width: 70em;
	}

	.story {
		display: block;
		width: 100%;
		border: none;
		margin-bottom: 3rem;
	}

	.story a {
		display: block;
		font-weight: 700;
		text-decoration: none;
	}

	.story span {
		display: block;
		margin-top: 1em;
		line-height: 1.2;
	}

	.wordmark {
		max-width: 10em;
		margin: 1em auto;
	}

	.about {
		margin: 3rem auto;
		margin-top: 0;
		text-align: center;
	}

	.links ul {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
	}

	.links li {
		display: flex;
		padding: 0.5em 1em;
	}

	.links a {
		display: flex;
		border: none;
		align-items: center;
		text-decoration: none;
	}

	.links span {
		margin-left: 0.5em;
	}

	@media only screen and (min-width: 30em) {
		.story {
			width: 50%;
			padding: 0 1em;
		}
	}

	@media only screen and (min-width: 50em) {
		.story {
			width: 25%;
			padding: 0 1em;
		}
	}
</style>
