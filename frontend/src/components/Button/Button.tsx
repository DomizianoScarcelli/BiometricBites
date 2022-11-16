import React from "react"

import "./Button.scss"

type ButtonProps = {
	text: string
	img: string
	shadow: boolean
}

const Button = ({ text, img, shadow }: ButtonProps) => {
	return (
		<>
			<button style={shadow ? { boxShadow: "4px 5px 19px rgba(var(--shadow-color-rgb), 0.5)" } : {}}>
				<p>
					{/* Strange trick to use \n in the text props */}
					{text.split("\n").map((str) => (
						<p>{str}</p>
					))}
				</p>
				<img alt="face emoji" src={img}></img>
			</button>
		</>
	)
}

export default Button
