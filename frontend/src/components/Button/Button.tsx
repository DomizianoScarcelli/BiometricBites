import React from "react"

import "./Button.scss"

type ButtonProps = {
	text: string
	img?: string
	shadow: boolean
	onClick: (event: React.MouseEvent<HTMLButtonElement>) => void
}

const Button = ({ text, img, shadow, onClick }: ButtonProps) => {
	return (
		<>
			<button style={shadow ? { boxShadow: "4px 5px 19px rgba(var(--shadow-color-rgb), 0.5)" } : {}} onClick={onClick}>
				<p>
					{/* Strange trick to use \n in the text props */}
					{text.split("\n").map((str, index) => (
						<p key={"row"+index}>{str}</p>
					))}
				</p>
				{img ? <img alt={`${img}`} src={img}></img> : <></>}
			</button>
		</>
	)
}

export default Button
