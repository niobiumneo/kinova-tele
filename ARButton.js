class ARButton {

	static createButton( renderer, sessionInit = {} ) {

		const button = document.createElement( 'button' );

		function stylizeElement( element ) {

			element.style.position = 'absolute';
			element.style.bottom = '20px';
			element.style.padding = '12px 6px';
			element.style.border = '1px solid #fff';
			element.style.borderRadius = '4px';
			element.style.background = 'rgba(0,0,0,0.25)';
			element.style.color = '#fff';
			element.style.font = 'normal 13px sans-serif';
			element.style.textAlign = 'center';
			element.style.opacity = '0.8';
			element.style.outline = 'none';
			element.style.zIndex = '999';

		}

		function disableButton() {

			button.style.display = '';
			button.style.cursor = 'auto';
			button.style.left = 'calc(50% - 90px)';
			button.style.width = '180px';
			button.onmouseenter = null;
			button.onmouseleave = null;
			button.onclick = null;

		}

		function showEnterAR() {

			let currentSession = null;

			async function onSessionStarted( session ) {

				session.addEventListener( 'end', onSessionEnded );
				await renderer.xr.setSession( session );
				button.textContent = 'EXIT PASSTHROUGH';
				currentSession = session;

			}

			function onSessionEnded() {

				currentSession.removeEventListener( 'end', onSessionEnded );
				button.textContent = 'ENTER PASSTHROUGH';
				currentSession = null;

			}

			button.style.display = '';
			button.style.cursor = 'pointer';
			button.style.left = 'calc(50% - 85px)';
			button.style.width = '170px';
			button.textContent = 'ENTER PASSTHROUGH';

			button.onmouseenter = function () {

				button.style.opacity = '1.0';

			};

			button.onmouseleave = function () {

				button.style.opacity = '0.8';

			};

			button.onclick = async function () {

				if ( currentSession !== null ) {

					await currentSession.end();
					return;

				}

				try {

					const session = await navigator.xr.requestSession( 'immersive-ar', sessionInit );
					await onSessionStarted( session );

				} catch ( error ) {

					console.error( 'Unable to start immersive-ar passthrough session', error );
					button.textContent = 'PASSTHROUGH FAILED';
					setTimeout( () => {

						if ( currentSession === null ) button.textContent = 'ENTER PASSTHROUGH';

					}, 2000 );

				}

			};

		}

		stylizeElement( button );

		if ( 'xr' in navigator ) {

			button.id = 'ARButton';
			button.style.display = 'none';

			navigator.xr.isSessionSupported( 'immersive-ar' ).then( function ( supported ) {

				if ( supported ) {

					showEnterAR();

				} else {

					disableButton();
					button.textContent = 'PASSTHROUGH NOT SUPPORTED';

				}

			} ).catch( function ( error ) {

				console.warn( 'Unable to check immersive-ar support', error );
				disableButton();
				button.textContent = 'PASSTHROUGH NOT ALLOWED';

			} );

			return button;

		}

		const message = document.createElement( 'a' );

		if ( window.isSecureContext === false ) {

			message.href = document.location.href.replace( /^http:/, 'https:' );
			message.textContent = 'WEBXR NEEDS HTTPS';

		} else {

			message.href = 'https://immersiveweb.dev/';
			message.textContent = 'WEBXR NOT AVAILABLE';

		}

		message.style.left = 'calc(50% - 90px)';
		message.style.width = '180px';
		message.style.textDecoration = 'none';
		stylizeElement( message );

		return message;

	}

}

export { ARButton };
